//! CueHammer Bridge — a VST3/CLAP plugin that shows up in CueHammer as an
//! output device (driver "VST Bridge"). The host's audio callback is the
//! clock: each `process()` consumes from a ring buffer fed over UDP by the
//! app and reports consumption, which is the app's signal to render more.
//! See `bridge-proto` for the wire protocol.

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use bridge_proto::{Packet, DISCOVERY_PORT};
use nih_plug::prelude::*;
use nih_plug_egui::{create_egui_editor, egui, EguiState};

const SESSION_TIMEOUT: Duration = Duration::from_secs(3);
const BEACON_INTERVAL: Duration = Duration::from_secs(1);
/// How often an unfilled hole is re-requested from the app (a Nack or its
/// resend can be lost too). LAN round trips are ~1 ms.
const NACK_INTERVAL: Duration = Duration::from_millis(5);
/// Net thread recv timeout: also the tick for re-nacks and give-ups.
const NET_TICK: Duration = Duration::from_millis(5);
/// Ring headroom beyond twice the host's declared max block; must exceed the
/// largest app-side extra-buffer setting (1000 ms) so the top-up target
/// (2×block + extra) always fits and RingWriter never has to drop frames.
const RING_SECONDS: f32 = 1.5;

pub struct Bridge {
    params: Arc<BridgeParams>,
    shared: Arc<Shared>,
    net: Option<NetThread>,
    consumer: Option<rtrb::Consumer<f32>>,
    /// Socket clone the audio thread sends `Consumed` on.
    audio_socket: Option<UdpSocket>,
    scratch: Vec<u8>,
    channels: usize,
    sample_rate: u32,
    /// Largest block the host has actually delivered since activation; the
    /// auto buffer target is twice this.
    max_block_frames: u32,
    /// The app has filled the ring at least once this session, so a short
    /// block is a real underrun rather than the handshake's empty ring.
    primed: bool,
    in_underrun: bool,
    /// Random per-plugin-object nonce, NOT persisted: lets the app tell a
    /// duplicated state chunk (two live objects, same instance id) from the
    /// same instance re-announcing after a rebind.
    boot_id: u64,
}

/// State shared between the audio thread, the network thread and the editor.
struct Shared {
    session: Mutex<Option<Session>>,
    /// Lock-free copy of the session for the audio thread: token (0 = no
    /// session) and the controller's IPv4 address packed as `ip << 16 |
    /// port`. The net thread republishes on every session change so
    /// `process()` never touches the mutex (a skipped Consumed under
    /// contention meant a late top-up).
    pub_token: AtomicU64,
    pub_addr: AtomicU64,
    /// Total frames output since activation; the stream timeline.
    consumed: AtomicU64,
    fill_frames: AtomicU32,
    sample_rate: AtomicU32,
    /// Blocks the ring could not fully serve (episodes, not blocks) since
    /// activation; the net thread reports changes to the app.
    underruns: AtomicU32,
    /// Largest host block seen; the net thread gives up waiting for a
    /// resend once the ring holds less than this.
    block_frames: AtomicU32,
}

struct Session {
    addr: SocketAddr,
    token: u64,
    last_rx: Instant,
}

/// Mirror the session (or its absence) into the audio thread's atomics.
fn publish_session(shared: &Shared, session: Option<&Session>) {
    match session {
        Some(s) => {
            let packed = match s.addr {
                SocketAddr::V4(a) => ((u32::from(*a.ip()) as u64) << 16) | a.port() as u64,
                // The socket is bound to an IPv4 address, so peers are V4.
                SocketAddr::V6(_) => 0,
            };
            shared.pub_addr.store(packed, Ordering::Relaxed);
            shared.pub_token.store(s.token, Ordering::Release);
        }
        None => {
            shared.pub_token.store(0, Ordering::Release);
            shared.pub_addr.store(0, Ordering::Relaxed);
        }
    }
}

fn unpack_addr(packed: u64) -> SocketAddr {
    SocketAddr::from((Ipv4Addr::from((packed >> 16) as u32), packed as u16))
}

/// macOS picks the wifi WMM queue from the socket's service type, not from
/// IP_TOS: mark the socket as interactive voice so Consumed (the app's pull
/// signal) and nacks get airtime priority over bulk traffic on this
/// machine's radio.
#[cfg(target_os = "macos")]
fn set_voice_service_type(socket: &UdpSocket) {
    use std::os::fd::AsRawFd;
    const SO_NET_SERVICE_TYPE: libc::c_int = 0x1116;
    const NET_SERVICE_TYPE_VO: libc::c_int = 4;
    // SAFETY: setsockopt on a live fd with a correctly sized c_int.
    unsafe {
        libc::setsockopt(
            socket.as_raw_fd(),
            libc::SOL_SOCKET,
            SO_NET_SERVICE_TYPE,
            &NET_SERVICE_TYPE_VO as *const libc::c_int as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        );
    }
}

struct NetThread {
    run: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
}

/// No DAW parameters at all: the name and the buffer leeway are both set
/// from CueHammer (typing into plugin GUIs is host-flaky, and one place to
/// configure beats two). Only identity and window size persist here.
#[derive(Params)]
struct BridgeParams {
    /// Instance name shown in CueHammer's device list; set via `SetName`.
    #[persist = "name"]
    name: RwLock<String>,
    /// Stable identity so a reloaded DAW project re-pairs to the same
    /// CueHammer device.
    #[persist = "instance_id"]
    instance_id: RwLock<String>,
    #[persist = "editor_state"]
    editor_state: Arc<EguiState>,
}

impl Default for BridgeParams {
    fn default() -> Self {
        let id = uuid::Uuid::new_v4();
        Self {
            // Default name = the readable handle of the id, so freshly
            // inserted instances are tellable apart without renaming.
            name: RwLock::new(bridge_proto::petname(id.as_bytes())),
            instance_id: RwLock::new(id.to_string()),
            editor_state: EguiState::from_size(320, 150),
        }
    }
}

impl Default for Bridge {
    fn default() -> Self {
        Self {
            params: Arc::new(BridgeParams::default()),
            shared: Arc::new(Shared {
                session: Mutex::new(None),
                pub_token: AtomicU64::new(0),
                pub_addr: AtomicU64::new(0),
                consumed: AtomicU64::new(0),
                fill_frames: AtomicU32::new(0),
                sample_rate: AtomicU32::new(0),
                underruns: AtomicU32::new(0),
                block_frames: AtomicU32::new(0),
            }),
            net: None,
            consumer: None,
            audio_socket: None,
            scratch: Vec::with_capacity(64),
            channels: 2,
            sample_rate: 0,
            max_block_frames: 0,
            primed: false,
            in_underrun: false,
            boot_id: uuid::Uuid::new_v4().as_u128() as u64,
        }
    }
}

impl Bridge {
    fn stop_net(&mut self) {
        if let Some(net) = self.net.take() {
            net.run.store(false, Ordering::Relaxed);
            let _ = net.handle.join();
        }
        // Tell the controller right away instead of letting it find out
        // from the 5 s beacon timeout: its backup output takes over sooner.
        if let (Some(socket), Some(session)) = (
            self.audio_socket.as_ref(),
            self.shared.session.lock().unwrap().as_ref(),
        ) {
            Packet::Bye {
                token: session.token,
            }
            .encode(&mut self.scratch);
            let _ = socket.send_to(&self.scratch, session.addr);
        }
        self.consumer = None;
        self.audio_socket = None;
        *self.shared.session.lock().unwrap() = None;
        publish_session(&self.shared, None);
        self.shared.consumed.store(0, Ordering::Relaxed);
        self.shared.fill_frames.store(0, Ordering::Relaxed);
        self.shared.underruns.store(0, Ordering::Relaxed);
        self.primed = false;
        self.in_underrun = false;
    }
}

const fn layout(ch: u32) -> AudioIOLayout {
    AudioIOLayout {
        main_input_channels: NonZeroU32::new(ch),
        main_output_channels: NonZeroU32::new(ch),
        aux_input_ports: &[],
        aux_output_ports: &[],
        names: PortNames::const_default(),
    }
}

impl Plugin for Bridge {
    const NAME: &'static str = "CueHammer Bridge";
    const VENDOR: &'static str = "Ondra Michalik";
    const URL: &'static str = "https://cuehammer.com";
    const EMAIL: &'static str = "ondra.michalik@gmail.com";
    const VERSION: &'static str = env!("BRIDGE_VERSION_DISPLAY");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] =
        &[layout(2), layout(1), layout(4), layout(8), layout(16)];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Box<dyn Editor>> {
        // Matches CueHammer's dark theme.
        const BG: egui::Color32 = egui::Color32::from_rgb(0x27, 0x27, 0x27);
        const PANEL: egui::Color32 = egui::Color32::from_rgb(0x36, 0x36, 0x36);
        const TEXT: egui::Color32 = egui::Color32::from_rgb(0xe0, 0xe0, 0xe0);
        const MUTED: egui::Color32 = egui::Color32::from_rgb(0x6e, 0x6e, 0x6e);
        const BORDER: egui::Color32 = egui::Color32::from_rgb(0x50, 0x50, 0x50);
        const ORANGE: egui::Color32 = egui::Color32::from_rgb(0xff, 0x95, 0x00);
        const GREEN: egui::Color32 = egui::Color32::from_rgb(0x4c, 0xaf, 0x50);

        let params = self.params.clone();
        let shared = self.shared.clone();
        // Editor state: the displayed fill in ms, time-averaged so the
        // per-block sawtooth of the raw ring fill reads as one number.
        create_egui_editor(
            self.params.editor_state.clone(),
            0.0f32,
            |ctx, _| {
                let mut style = (*ctx.style()).clone();
                style.visuals = egui::Visuals::dark();
                style.visuals.panel_fill = BG;
                style.visuals.window_fill = BG;
                style.visuals.override_text_color = Some(TEXT);
                style.visuals.selection.bg_fill = ORANGE;
                ctx.set_style(style);
            },
            move |ctx, _setter, avg: &mut f32| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::default().fill(BG).inner_margin(14.0))
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("CueHammer")
                                    .size(17.0)
                                    .strong()
                                    .color(TEXT),
                            );
                            ui.label(egui::RichText::new("Bridge").size(17.0).color(ORANGE));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(
                                        egui::RichText::new(Bridge::VERSION)
                                            .size(10.0)
                                            .color(MUTED),
                                    );
                                },
                            );
                        });
                        ui.add_space(10.0);

                        egui::Frame::default()
                            .fill(PANEL)
                            .stroke(egui::Stroke::new(1.0_f32, BORDER))
                            .corner_radius(3.0)
                            .inner_margin(10.0)
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("Name").color(MUTED));
                                    ui.label(
                                        egui::RichText::new(params.name.read().unwrap().clone())
                                            .size(14.0)
                                            .strong()
                                            .color(TEXT),
                                    );
                                });
                                ui.add_space(6.0);
                                let session = shared.session.lock().unwrap();
                                let sr = shared.sample_rate.load(Ordering::Relaxed);
                                let fill = shared.fill_frames.load(Ordering::Relaxed);
                                let fill_ms = if sr > 0 {
                                    fill as f32 * 1000.0 / sr as f32
                                } else {
                                    0.0
                                };
                                // ponytail: EMA with a ~1 s time constant; a
                                // windowed mean only if min/max are ever shown.
                                let dt = ctx.input(|i| i.stable_dt).min(0.1);
                                *avg += (fill_ms - *avg) * dt;
                                let (color, text) = match (&*session, sr) {
                                    (_, 0) => (MUTED, "Inactive".to_string()),
                                    (Some(_), _) => {
                                        (GREEN, format!("Connected — {avg:.0} ms buffered"))
                                    }
                                    (None, _) => (ORANGE, "Waiting for CueHammer".to_string()),
                                };
                                if session.is_none() || sr == 0 {
                                    *avg = 0.0;
                                }
                                ui.horizontal(|ui| {
                                    ui.label(egui::RichText::new("●").color(color));
                                    ui.label(egui::RichText::new(text).color(color));
                                });
                            });

                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(
                                "Name and buffer are set in CueHammer's Outputs tab.",
                            )
                            .size(11.0)
                            .color(MUTED),
                        );
                    });
            },
        )
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.stop_net();

        self.channels = audio_io_layout
            .main_output_channels
            .map(NonZeroU32::get)
            .unwrap_or(2) as usize;
        self.sample_rate = buffer_config.sample_rate as u32;
        self.max_block_frames = 0;
        self.shared
            .sample_rate
            .store(self.sample_rate, Ordering::Relaxed);

        let capacity = ((self.sample_rate as f32 * RING_SECONDS) as usize
            + buffer_config.max_buffer_size as usize * 2)
            * self.channels;
        let (producer, consumer) = rtrb::RingBuffer::new(capacity);
        self.consumer = Some(consumer);

        let socket = match UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0)) {
            Ok(s) => s,
            Err(err) => {
                nih_error!("bridge: failed to bind socket: {err}");
                return false;
            }
        };
        let _ = socket.set_broadcast(true);
        let _ = socket.set_read_timeout(Some(NET_TICK));
        // A top-up burst is up to 4 host blocks × 16 ch of f32, and the OS
        // default receive buffer (64 KB on Windows) holds ~20 ms of that:
        // a net thread scheduled late would lose the tail to nacks.
        let _ = socket2::SockRef::from(&socket).set_recv_buffer_size(4 << 20);
        #[cfg(target_os = "macos")]
        set_voice_service_type(&socket);
        let audio_socket = match socket.try_clone() {
            Ok(s) => {
                let _ = s.set_nonblocking(true);
                s
            }
            Err(err) => {
                nih_error!("bridge: failed to clone socket: {err}");
                return false;
            }
        };
        self.audio_socket = Some(audio_socket);

        let run = Arc::new(AtomicBool::new(true));
        let ctx = NetContext {
            socket,
            producer,
            shared: self.shared.clone(),
            params: self.params.clone(),
            run: run.clone(),
            channels: self.channels as u16,
            sample_rate: self.sample_rate,
            boot_id: self.boot_id,
        };
        let handle = std::thread::Builder::new()
            .name("bridge-net".to_string())
            .spawn(move || net_thread(ctx))
            .expect("spawn bridge net thread");
        self.net = Some(NetThread { run, handle });
        true
    }

    fn deactivate(&mut self) {
        self.stop_net();
        self.shared.sample_rate.store(0, Ordering::Relaxed);
    }

    fn reset(&mut self) {}

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let frames = buffer.samples();
        let channels = self.channels;
        let output = buffer.as_slice();
        // Output only what CueHammer sends: the host's input is not passed
        // through, and frames the ring can't supply are silence.
        for ch in output.iter_mut() {
            ch.fill(0.0);
        }

        if let Some(consumer) = self.consumer.as_mut() {
            let avail = consumer.slots();
            let want = frames * channels;
            let short = avail < want;
            // Episodes, not blocks; zero-frame flush calls carry no
            // information and must not prime or end an episode.
            if want > 0 {
                if short {
                    if self.primed && !self.in_underrun {
                        self.shared.underruns.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    self.primed = true;
                }
                self.in_underrun = short;
            }
            let take = avail.min(want) / channels * channels;
            if take > 0 {
                if let Ok(chunk) = consumer.read_chunk(take) {
                    let (a, b) = chunk.as_slices();
                    for (i, s) in a.iter().chain(b.iter()).enumerate() {
                        output[i % channels][i / channels] = *s;
                    }
                    chunk.commit_all();
                }
            }

            let consumed = self
                .shared
                .consumed
                .fetch_add(frames as u64, Ordering::Relaxed)
                + frames as u64;
            let fill = (consumer.slots() / channels) as u32;
            self.shared.fill_frames.store(fill, Ordering::Relaxed);

            // Reported target = the automatic minimum only: two of the
            // largest block the host has actually used (not the often much
            // larger max it declared; adapts up, never down, so a one-off
            // small block can't shrink it). The user's extra leeway is
            // configured in CueHammer and added on the app side.
            self.max_block_frames = self.max_block_frames.max(frames as u32);
            let target_frames = self.max_block_frames * 2;
            self.shared
                .block_frames
                .store(self.max_block_frames, Ordering::Relaxed);

            // Lock-free: the net thread mirrors the session into atomics.
            // A stale token/addr pair during a controller swap is harmless,
            // the app drops packets with the wrong token.
            let token = self.shared.pub_token.load(Ordering::Acquire);
            if token == 0 {
                // No controller: the ring draining is not an underrun.
                self.primed = false;
            } else if let Some(socket) = self.audio_socket.as_ref() {
                let addr = unpack_addr(self.shared.pub_addr.load(Ordering::Relaxed));
                Packet::Consumed {
                    token,
                    stream_frame: consumed,
                    fill_frames: fill,
                    target_frames,
                }
                .encode(&mut self.scratch);
                let _ = socket.send_to(&self.scratch, addr);
            }
        }

        ProcessStatus::KeepAlive
    }
}

struct NetContext {
    socket: UdpSocket,
    producer: rtrb::Producer<f32>,
    shared: Arc<Shared>,
    params: Arc<BridgeParams>,
    run: Arc<AtomicBool>,
    channels: u16,
    sample_rate: u32,
    boot_id: u64,
}

/// Where beacons go: limited broadcast, loopback, and every interface's own
/// subnet broadcast — 255.255.255.255 leaves on the default-route interface
/// only, which is the internet-facing wifi on a typical FOH laptop, not the
/// audio LAN. Re-enumerated per beacon so a cable plugged in later counts.
fn beacon_targets() -> Vec<Ipv4Addr> {
    let mut targets = vec![Ipv4Addr::BROADCAST, Ipv4Addr::LOCALHOST];
    for iface in get_if_addrs::get_if_addrs().unwrap_or_default() {
        if let get_if_addrs::IfAddr::V4(v4) = iface.addr {
            if let Some(b) = v4.broadcast.filter(|b| !targets.contains(b)) {
                targets.push(b);
            }
        }
    }
    targets
}

/// Session check shared by the packets that carry audio: right token, right
/// controller, and it refreshes the session timeout.
fn accept_audio(shared: &Shared, token: u64, from: SocketAddr) -> bool {
    let mut session = shared.session.lock().unwrap();
    match session.as_mut() {
        Some(s) if s.token == token && s.addr == from => {
            s.last_rx = Instant::now();
            true
        }
        _ => false,
    }
}

fn net_thread(ctx: NetContext) {
    // This thread feeds the audio thread's ring: a late wakeup under a busy
    // DAW GUI is an underrun. Same footing as the app's render thread.
    let _ = thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max);

    let mut instance_id = uuid::Uuid::parse_str(&ctx.params.instance_id.read().unwrap())
        .map(|u| *u.as_bytes())
        .unwrap_or_default();
    let listen_port = ctx.socket.local_addr().map(|a| a.port()).unwrap_or(0);
    let mut writer = RingWriter {
        producer: ctx.producer,
        next_frame: 0,
        channels: ctx.channels as usize,
        gaps: 0,
        stash: BTreeMap::new(),
    };
    let mut out = Vec::with_capacity(128);
    let mut buf = [0u8; 2048];
    // Zero source for `Silence`; grows to the largest run seen, never shrinks.
    let mut zeros: Vec<f32> = Vec::new();
    let mut last_beacon = Instant::now() - BEACON_INTERVAL;
    let mut last_nack = Instant::now();
    let mut reported_gaps = 0u32;
    let mut reported_underruns = 0u32;

    while ctx.run.load(Ordering::Relaxed) {
        // Underruns are counted on the audio thread; report changes from
        // here, at worst one recv timeout (NET_TICK) late.
        let underruns = ctx.shared.underruns.load(Ordering::Relaxed);
        if underruns != reported_underruns {
            let session = ctx.shared.session.lock().unwrap();
            if let Some(s) = &*session {
                reported_underruns = underruns;
                Packet::Underruns {
                    token: s.token,
                    total: underruns,
                }
                .encode(&mut out);
                let _ = ctx.socket.send_to(&out, s.addr);
            }
        }
        if last_beacon.elapsed() >= BEACON_INTERVAL {
            last_beacon = Instant::now();
            Packet::Beacon {
                instance_id,
                boot_id: ctx.boot_id,
                listen_port,
                channels: ctx.channels,
                sample_rate: ctx.sample_rate,
                name: ctx.params.name.read().unwrap().clone(),
            }
            .encode(&mut out);
            for ip in beacon_targets() {
                let _ = ctx.socket.send_to(&out, (ip, DISCOVERY_PORT));
            }

            let mut session = ctx.shared.session.lock().unwrap();
            if session
                .as_ref()
                .is_some_and(|s| s.last_rx.elapsed() > SESSION_TIMEOUT)
            {
                *session = None;
                publish_session(&ctx.shared, None);
            }
        }

        let received = ctx.socket.recv_from(&mut buf);

        // Holes waiting on a resend: give up on those the ring can no
        // longer afford to wait for (silence + gap), re-request the rest.
        if !writer.stash.is_empty() {
            writer.give_up(
                ctx.shared.consumed.load(Ordering::Relaxed),
                ctx.shared.block_frames.load(Ordering::Relaxed) as u64,
            );
            if writer.gaps != reported_gaps {
                reported_gaps = writer.gaps;
                let session = ctx.shared.session.lock().unwrap();
                if let Some(s) = &*session {
                    Packet::Gaps {
                        token: s.token,
                        total: writer.gaps,
                    }
                    .encode(&mut out);
                    let _ = ctx.socket.send_to(&out, s.addr);
                }
            }
            if !writer.stash.is_empty() && last_nack.elapsed() >= NACK_INTERVAL {
                last_nack = Instant::now();
                let session = ctx.shared.session.lock().unwrap();
                if let Some(s) = &*session {
                    for (start_frame, end_frame) in writer.holes() {
                        Packet::Nack {
                            token: s.token,
                            start_frame,
                            end_frame,
                        }
                        .encode(&mut out);
                        let _ = ctx.socket.send_to(&out, s.addr);
                    }
                }
            }
        }

        let (len, from) = match received {
            Ok(ok) => ok,
            Err(_) => continue,
        };
        let Some(packet) = Packet::decode(&buf[..len]) else {
            continue;
        };
        match packet {
            Packet::Hello { token } => {
                let mut session = ctx.shared.session.lock().unwrap();
                // First controller wins; the same one may refresh or rebind.
                let free = match &*session {
                    None => true,
                    Some(s) => s.addr == from || s.token == token,
                };
                if free {
                    // A new controller session counts gaps from zero, matching
                    // its own baseline; a same-token refresh keeps the total.
                    if session.as_ref().map_or(true, |s| s.token != token) {
                        writer.gaps = 0;
                        writer.stash.clear();
                        reported_gaps = 0;
                        ctx.shared.underruns.store(0, Ordering::Relaxed);
                        reported_underruns = 0;
                    }
                    *session = Some(Session {
                        addr: from,
                        token,
                        last_rx: Instant::now(),
                    });
                    publish_session(&ctx.shared, session.as_ref());
                    drop(session);
                    Packet::Welcome {
                        token,
                        channels: ctx.channels,
                        sample_rate: ctx.sample_rate,
                    }
                    .encode(&mut out);
                    let _ = ctx.socket.send_to(&out, from);
                }
            }
            Packet::Audio {
                token,
                start_frame,
                channels,
                samples,
            } => {
                if channels != ctx.channels || !accept_audio(&ctx.shared, token, from) {
                    continue;
                }
                // A new hole: nack it as soon as the loop comes round (the
                // next packet, or at worst one NET_TICK), not a full
                // NACK_INTERVAL later.
                if writer.write(start_frame, &samples, &ctx.shared.consumed) {
                    last_nack = Instant::now() - NACK_INTERVAL;
                }
            }
            Packet::Silence {
                token,
                start_frame,
                frames,
            } => {
                if !accept_audio(&ctx.shared, token, from) {
                    continue;
                }
                let n = frames as usize * ctx.channels as usize;
                if zeros.len() < n {
                    zeros.resize(n, 0.0);
                }
                if writer.write(start_frame, &zeros[..n], &ctx.shared.consumed) {
                    last_nack = Instant::now() - NACK_INTERVAL;
                }
            }
            Packet::Bye { token } => {
                let mut session = ctx.shared.session.lock().unwrap();
                if session.as_ref().is_some_and(|s| s.token == token) {
                    *session = None;
                    publish_session(&ctx.shared, None);
                }
            }
            Packet::SetName {
                instance_id: id,
                name,
            } => {
                if id == instance_id && !name.is_empty() {
                    *ctx.params.name.write().unwrap() = name;
                    // Re-beacon now so the app's device list updates at once.
                    last_beacon = Instant::now() - BEACON_INTERVAL;
                }
            }
            Packet::Reassign {
                instance_id: id,
                new_id,
            } => {
                // We are a duplicated copy; adopt the fresh identity (and
                // its default petname, so the twins are tellable apart).
                // Persisted with the next DAW project save.
                if id == instance_id {
                    instance_id = new_id;
                    *ctx.params.instance_id.write().unwrap() =
                        uuid::Uuid::from_bytes(new_id).to_string();
                    *ctx.params.name.write().unwrap() = bridge_proto::petname(&new_id);
                    last_beacon = Instant::now() - BEACON_INTERVAL;
                }
            }
            _ => {}
        }
    }
}

struct RingWriter {
    producer: rtrb::Producer<f32>,
    /// Next absolute frame position to be written to the ring.
    next_frame: u64,
    channels: usize,
    /// Holes given up on and filled with silence since activation.
    gaps: u32,
    /// Packets that arrived ahead of a hole, keyed by start frame. They wait
    /// for the app's resend and are pushed the moment the hole closes.
    stash: BTreeMap<u64, Vec<f32>>,
}

impl RingWriter {
    /// Write interleaved samples starting at an absolute frame position.
    /// Late samples are dropped; a packet past a hole is stashed (returns
    /// true: the hole should be nacked) instead of silence-filling the hole;
    /// after an underrun the position fast-forwards to the consumed clock so
    /// stale silence never adds latency.
    fn write(&mut self, start_frame: u64, samples: &[f32], consumed: &AtomicU64) -> bool {
        let consumed = consumed.load(Ordering::Relaxed);
        if self.next_frame < consumed {
            self.next_frame = consumed;
        }
        if start_frame > self.next_frame {
            self.stash.insert(start_frame, samples.to_vec());
            return true;
        }
        self.push_at(start_frame, samples);
        self.drain_stash();
        false
    }

    /// Push samples at a position at or before `next_frame`, trimming the
    /// already-covered head. Returns false if the ring came up short.
    fn push_at(&mut self, start_frame: u64, samples: &[f32]) -> bool {
        let ch = self.channels;
        let frames = samples.len() / ch;
        if start_frame + frames as u64 <= self.next_frame {
            return true;
        }
        let skip = (self.next_frame - start_frame) as usize * ch;
        let samples = &samples[skip..];
        self.push(&mut samples.iter().copied(), samples.len()) == samples.len()
    }

    /// Push every stashed packet that now touches `next_frame`.
    // ponytail: a packet the full ring only half takes loses its tail; the
    // gap that leaves is nacked and resent, and RING_HEADROOM is sized so
    // the ring never fills. Reinsert the remainder if that ever changes.
    fn drain_stash(&mut self) {
        while let Some(entry) = self.stash.first_entry() {
            if *entry.key() > self.next_frame {
                return;
            }
            let (start, samples) = entry.remove_entry();
            if !self.push_at(start, &samples) {
                return;
            }
        }
    }

    /// Frame ranges the stash is waiting on, oldest first.
    fn holes(&self) -> Vec<(u64, u64)> {
        let mut holes = Vec::new();
        let mut prev = self.next_frame;
        for (start, samples) in &self.stash {
            if *start > prev {
                holes.push((prev, *start));
            }
            prev = prev.max(start + (samples.len() / self.channels) as u64);
        }
        holes
    }

    /// Stop waiting for resends the ring can no longer afford: while less
    /// than one host block is buffered, silence-fill the oldest hole and
    /// count it as a gap.
    // ponytail: threshold = one host block, checked once per NET_TICK, so a
    // host with blocks shorter than the tick underruns before give-up fires
    // (counted as an underrun, not a gap). Make it a setting if a resend
    // ever needs more than that.
    fn give_up(&mut self, consumed: u64, block: u64) {
        // Same clamp as `write`: after an underrun the past is gone, so a
        // hole (and any stash) below `consumed` must not be pushed as stale
        // latency.
        if self.next_frame < consumed {
            self.next_frame = consumed;
        }
        loop {
            self.drain_stash();
            let Some(&first) = self.stash.keys().next() else {
                return;
            };
            if self.next_frame.saturating_sub(consumed) >= block {
                return;
            }
            self.gaps += 1;
            let gap = (first - self.next_frame) as usize * self.channels;
            if self.push(&mut std::iter::repeat(0.0f32), gap) < gap {
                return;
            }
        }
    }

    /// Push up to `want` samples (floored to whole frames) from the
    /// iterator; returns how many were written and advances `next_frame`.
    fn push(&mut self, samples: &mut impl Iterator<Item = f32>, want: usize) -> usize {
        let n = self.producer.slots().min(want) / self.channels * self.channels;
        if n == 0 {
            return 0;
        }
        let Ok(chunk) = self.producer.write_chunk_uninit(n) else {
            return 0;
        };
        let written = chunk.fill_from_iter(samples.take(n));
        self.next_frame += (written / self.channels) as u64;
        written
    }
}

impl ClapPlugin for Bridge {
    const CLAP_ID: &'static str = "cz.ondramichalik.cuehammer-bridge";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Receives CueHammer playback as a plugin in any host");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Utility];
}

impl Vst3Plugin for Bridge {
    const VST3_CLASS_ID: [u8; 16] = *b"CueHammerBridge!";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Network];
}

nih_export_clap!(Bridge);
nih_export_vst3!(Bridge);

#[cfg(test)]
mod tests {
    use super::*;

    fn writer(cap: usize) -> (RingWriter, rtrb::Consumer<f32>) {
        let (producer, consumer) = rtrb::RingBuffer::new(cap);
        (
            RingWriter {
                producer,
                next_frame: 0,
                channels: 2,
                gaps: 0,
                stash: BTreeMap::new(),
            },
            consumer,
        )
    }

    fn drain(consumer: &mut rtrb::Consumer<f32>) -> Vec<f32> {
        let mut out = Vec::new();
        while let Ok(s) = consumer.pop() {
            out.push(s);
        }
        out
    }

    #[test]
    fn ring_stashes_and_recovers() {
        let (mut w, mut c) = writer(64);
        let consumed = AtomicU64::new(0);
        assert!(!w.write(0, &[1.0, 2.0], &consumed));
        // Frame 2 with frame 1 missing: held back, hole reported, no silence.
        assert!(w.write(2, &[5.0, 6.0], &consumed));
        assert_eq!(w.holes(), vec![(1, 2)]);
        assert_eq!(drain(&mut c), vec![1.0, 2.0]);
        assert_eq!(w.gaps, 0);
        // The resend closes the hole and the stash follows in order.
        assert!(!w.write(1, &[3.0, 4.0], &consumed));
        assert_eq!(drain(&mut c), vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(w.next_frame, 3);
        assert!(w.stash.is_empty());
        assert_eq!(w.gaps, 0);
    }

    #[test]
    fn ring_gives_up_when_short() {
        let (mut w, mut c) = writer(64);
        let consumed = AtomicU64::new(0);
        w.write(0, &[1.0, 2.0], &consumed);
        w.write(2, &[5.0, 6.0], &consumed);
        // Plenty buffered relative to the block: keep waiting.
        w.give_up(0, 1);
        assert_eq!(w.gaps, 0);
        // Less than a block left: silence the hole and move on.
        w.give_up(0, 1000);
        assert_eq!(drain(&mut c), vec![1.0, 2.0, 0.0, 0.0, 5.0, 6.0]);
        assert_eq!(w.gaps, 1);
        assert!(w.stash.is_empty());
    }

    #[test]
    fn ring_give_up_skips_the_past_after_underrun() {
        let (mut w, mut c) = writer(64);
        let consumed = AtomicU64::new(0);
        w.write(0, &[1.0, 2.0], &consumed);
        w.write(2, &[5.0, 6.0], &consumed);
        drain(&mut c);
        // The host played through frame 5 while the hole was open: neither
        // the hole nor the stashed frame 2 may reach the ring as stale audio.
        w.give_up(6, 1000);
        assert_eq!(drain(&mut c), Vec::<f32>::new());
        assert_eq!(w.next_frame, 6);
        assert!(w.stash.is_empty());
        assert_eq!(w.gaps, 0);
    }

    #[test]
    fn ring_drops_late_and_partially_late() {
        let (mut w, mut c) = writer(64);
        let consumed = AtomicU64::new(0);
        w.write(0, &[1.0, 1.0, 2.0, 2.0], &consumed);
        // Fully late duplicate: dropped.
        w.write(0, &[9.0, 9.0], &consumed);
        // Overlapping: only the new tail (frame 2) lands.
        w.write(1, &[9.0, 9.0, 3.0, 3.0], &consumed);
        assert_eq!(drain(&mut c), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn ring_fast_forwards_after_underrun() {
        let (mut w, mut c) = writer(64);
        let consumed = AtomicU64::new(10);
        // The app answers an underrun by sending from the consumed clock;
        // stale positions must not become latency-adding silence.
        w.write(10, &[7.0, 8.0], &consumed);
        assert_eq!(w.next_frame, 11);
        assert_eq!(drain(&mut c), vec![7.0, 8.0]);
        // Anything older than the consumed clock is dropped entirely.
        w.write(5, &[9.0, 9.0], &consumed);
        assert_eq!(drain(&mut c), Vec::<f32>::new());
    }

    #[test]
    fn net_thread_session_loopback() {
        let shared = Arc::new(Shared {
            session: Mutex::new(None),
            pub_token: AtomicU64::new(0),
            pub_addr: AtomicU64::new(0),
            consumed: AtomicU64::new(0),
            fill_frames: AtomicU32::new(0),
            sample_rate: AtomicU32::new(48000),
            underruns: AtomicU32::new(0),
            block_frames: AtomicU32::new(0),
        });
        let (producer, mut consumer) = rtrb::RingBuffer::new(256);
        let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_millis(20)))
            .unwrap();
        let plugin_addr = socket.local_addr().unwrap();
        let run = Arc::new(AtomicBool::new(true));
        let handle = std::thread::spawn({
            let ctx = NetContext {
                socket,
                producer,
                shared: shared.clone(),
                params: Arc::new(BridgeParams::default()),
                run: run.clone(),
                channels: 2,
                sample_rate: 48000,
                boot_id: 1,
            };
            move || net_thread(ctx)
        });

        let app = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        app.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut out = Vec::new();
        Packet::Hello { token: 42 }.encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();

        let mut buf = [0u8; 2048];
        let welcome = loop {
            let (len, _) = app.recv_from(&mut buf).unwrap();
            // The beacon to localhost:DISCOVERY_PORT never lands here, but
            // skip anything that is not the Welcome just in case.
            if let Some(p @ Packet::Welcome { .. }) = Packet::decode(&buf[..len]) {
                break p;
            }
        };
        assert_eq!(
            welcome,
            Packet::Welcome {
                token: 42,
                channels: 2,
                sample_rate: 48000
            }
        );

        Packet::Audio {
            token: 42,
            start_frame: 0,
            channels: 2,
            samples: vec![0.25, -0.25],
        }
        .encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        while consumer.slots() < 2 {
            assert!(Instant::now() < deadline, "audio never reached the ring");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(drain(&mut consumer), vec![0.25, -0.25]);

        // Silence lands as zero frames on the same timeline.
        Packet::Silence {
            token: 42,
            start_frame: 1,
            frames: 1,
        }
        .encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while consumer.slots() < 2 {
            assert!(Instant::now() < deadline, "silence never reached the ring");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(drain(&mut consumer), vec![0.0, 0.0]);

        // Frame 3 without frame 2: the plugin asks for the hole instead of
        // padding it.
        Packet::Audio {
            token: 42,
            start_frame: 3,
            channels: 2,
            samples: vec![0.5, -0.5],
        }
        .encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();
        let nack = loop {
            let (len, _) = app.recv_from(&mut buf).expect("nack never arrived");
            if let Some(p @ Packet::Nack { .. }) = Packet::decode(&buf[..len]) {
                break p;
            }
        };
        assert_eq!(
            nack,
            Packet::Nack {
                token: 42,
                start_frame: 2,
                end_frame: 3
            }
        );
        assert_eq!(consumer.slots(), 0, "stashed audio must not reach the ring");

        Packet::Bye { token: 42 }.encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while shared.session.lock().unwrap().is_some() {
            assert!(Instant::now() < deadline, "session never cleared");
            std::thread::sleep(Duration::from_millis(5));
        }

        run.store(false, Ordering::Relaxed);
        handle.join().unwrap();
    }

    /// A Reassign packet makes the plugin adopt the fresh identity and its
    /// default petname — the duplicated-state-chunk recovery path.
    #[test]
    fn net_thread_adopts_reassigned_identity() {
        let params = Arc::new(BridgeParams::default());
        let old_id = *uuid::Uuid::parse_str(&params.instance_id.read().unwrap().clone())
            .unwrap()
            .as_bytes();
        let new_id = *uuid::Uuid::new_v4().as_bytes();

        let (producer, _consumer) = rtrb::RingBuffer::new(64);
        let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        socket
            .set_read_timeout(Some(Duration::from_millis(20)))
            .unwrap();
        let plugin_addr = socket.local_addr().unwrap();
        let run = Arc::new(AtomicBool::new(true));
        let handle = std::thread::spawn({
            let ctx = NetContext {
                socket,
                producer,
                shared: Arc::new(Shared {
                    session: Mutex::new(None),
                    pub_token: AtomicU64::new(0),
                    pub_addr: AtomicU64::new(0),
                    consumed: AtomicU64::new(0),
                    fill_frames: AtomicU32::new(0),
                    sample_rate: AtomicU32::new(48000),
                    underruns: AtomicU32::new(0),
                    block_frames: AtomicU32::new(0),
                }),
                params: params.clone(),
                run: run.clone(),
                channels: 2,
                sample_rate: 48000,
                boot_id: 7,
            };
            move || net_thread(ctx)
        });

        let app = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let mut out = Vec::new();
        // A Reassign for someone else's id must be ignored.
        Packet::Reassign {
            instance_id: [0xAA; 16],
            new_id: [0xBB; 16],
        }
        .encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();
        Packet::Reassign {
            instance_id: old_id,
            new_id,
        }
        .encode(&mut out);
        app.send_to(&out, plugin_addr).unwrap();

        let want_id = uuid::Uuid::from_bytes(new_id).to_string();
        let deadline = Instant::now() + Duration::from_secs(2);
        while *params.instance_id.read().unwrap() != want_id {
            assert!(Instant::now() < deadline, "identity never adopted");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(*params.name.read().unwrap(), bridge_proto::petname(&new_id));

        run.store(false, Ordering::Relaxed);
        handle.join().unwrap();
    }
}
