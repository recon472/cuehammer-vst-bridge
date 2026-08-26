//! Wire protocol between CueHammer and the Bridge plugin.
//!
//! The plugin broadcasts [`Packet::Beacon`] once a second so the app can list
//! instances as devices. The app claims an instance with [`Packet::Hello`].
//! From then on the plugin's audio callback is the clock: every `process()`
//! it sends [`Packet::Consumed`] and the app answers with enough
//! [`Packet::Audio`] to restore the target ring fill. All frame positions
//! share one timeline: total frames the plugin has output since activation.
//!
//! Everything is little-endian over UDP, one packet per datagram.

/// Beacon listener port on the app side. Next to the mobile remote's 27412,
/// clear of Dante (319/320, 4440-4455, 5353, 8700-8899, 14336+).
pub const DISCOVERY_PORT: u16 = 27413;

pub const VERSION: u8 = 1;
pub const MAX_NAME_BYTES: usize = 64;
/// Keep every datagram under a conservative Ethernet MTU.
pub const MAX_PACKET_BYTES: usize = 1400;

const MAGIC: [u8; 4] = *b"CHBR";
const HEADER_BYTES: usize = 6;
const AUDIO_OVERHEAD_BYTES: usize = HEADER_BYTES + 8 + 8 + 2 + 2;

/// How many frames fit in one `Audio` datagram for a channel count.
pub fn frames_per_packet(channels: u16) -> usize {
    ((MAX_PACKET_BYTES - AUDIO_OVERHEAD_BYTES) / (4 * channels.max(1) as usize)).max(1)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Packet {
    /// Plugin -> broadcast:DISCOVERY_PORT, ~1 Hz while active.
    Beacon {
        instance_id: [u8; 16],
        /// Random per-plugin-object nonce (not persisted). Two live objects
        /// beaconing the same instance_id with different boot_ids means the
        /// state chunk was duplicated (copied rack/channel) — the app then
        /// hands the newcomer a fresh identity via [`Packet::Reassign`].
        boot_id: u64,
        /// Port the plugin listens on for Hello/Audio/Bye.
        listen_port: u16,
        channels: u16,
        sample_rate: u32,
        name: String,
    },
    /// App -> plugin. First controller wins; repeats from the same address
    /// refresh the session.
    Hello { token: u64 },
    /// Plugin -> app, confirms the session and the stream format.
    Welcome {
        token: u64,
        channels: u16,
        sample_rate: u32,
    },
    /// Plugin -> app, once per audio callback. The pull signal.
    Consumed {
        token: u64,
        /// Total frames output since activation (ring content starts here).
        stream_frame: u64,
        /// Frames currently buffered in the plugin ring.
        fill_frames: u32,
        /// The plugin's automatic minimum fill (two host blocks). The app
        /// adds its configured per-instance leeway on top before topping up.
        target_frames: u32,
    },
    /// App -> plugin. Interleaved f32 starting at an absolute frame position.
    Audio {
        token: u64,
        start_frame: u64,
        channels: u16,
        samples: Vec<f32>,
    },
    /// Either direction; sessions also die by timeout.
    Bye { token: u64 },
    /// App -> plugin: rename the instance (typing into plugin GUIs is
    /// host-flaky, so the app's UI is the reliable place to edit the name).
    /// Addressed by instance id, no session needed; the plugin persists the
    /// name in its state chunk and re-beacons immediately.
    SetName { instance_id: [u8; 16], name: String },
    /// App -> plugin: adopt a fresh identity. Sent when two live instances
    /// beacon the same id (duplicated state chunk); the established one
    /// keeps the id, the newcomer takes `new_id` (and its default petname),
    /// persists it, and re-beacons as a new device.
    Reassign {
        instance_id: [u8; 16],
        new_id: [u8; 16],
    },
    /// Plugin -> app, whenever the count changes: total `Audio` gaps (lost or
    /// reordered packets the plugin filled with silence) since activation.
    /// Cumulative so a lost report is caught up by the next one; a separate
    /// kind so older peers just ignore it.
    Gaps { token: u64, total: u32 },
    /// Plugin -> app, whenever the count changes: total ring underruns (host
    /// blocks the plugin had to pad with silence) since activation. Counted
    /// in the audio callback, the only place that knows a block came up
    /// short — the app can't tell a late packet from one still in flight.
    Underruns { token: u64, total: u32 },
    /// Plugin -> app: frames `[start_frame, end_frame)` never arrived. The
    /// app resends them from its render history if it still has them; the
    /// plugin repeats the request until the hole fills or it has to give up
    /// (ring nearly dry) and pad with silence.
    Nack {
        token: u64,
        start_frame: u64,
        end_frame: u64,
    },
}

/// Human-readable handle derived from an instance id, e.g. "Amber-Fox-31".
/// Used as the default instance name so devices are tellable apart; identity
/// itself stays the full 16-byte id.
pub fn petname(id: &[u8; 16]) -> String {
    const ADJECTIVES: [&str; 32] = [
        "Amber", "Bold", "Calm", "Crimson", "Dusty", "Emerald", "Fast", "Gentle", "Golden",
        "Happy", "Icy", "Jade", "Keen", "Lucky", "Mellow", "Noble", "Olive", "Proud", "Quick",
        "Royal", "Rusty", "Silent", "Silver", "Smooth", "Solar", "Swift", "Tidal", "Violet",
        "Vivid", "Wild", "Witty", "Zesty",
    ];
    const ANIMALS: [&str; 32] = [
        "Badger", "Bear", "Bison", "Crane", "Crow", "Deer", "Dove", "Eagle", "Falcon", "Finch",
        "Fox", "Gecko", "Hawk", "Heron", "Ibis", "Koala", "Lemur", "Llama", "Lynx", "Marten",
        "Mole", "Moose", "Otter", "Owl", "Panda", "Puffin", "Raven", "Robin", "Seal", "Stork",
        "Tiger", "Wolf",
    ];
    format!(
        "{}-{}-{}",
        ADJECTIVES[(id[0] & 31) as usize],
        ANIMALS[(id[1] & 31) as usize],
        10 + (id[2] % 90)
    )
}

/// Name bytes capped to [`MAX_NAME_BYTES`], truncated on a char boundary so
/// the result stays valid UTF-8 (decode rejects invalid names wholesale).
fn name_bytes(name: &str) -> &[u8] {
    let mut end = name.len().min(MAX_NAME_BYTES);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name.as_bytes()[..end]
}

impl Packet {
    /// Serialize into `out` (cleared first). The buffer is reusable to keep
    /// the hot path allocation-free.
    pub fn encode(&self, out: &mut Vec<u8>) {
        out.clear();
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        match self {
            Packet::Beacon {
                instance_id,
                boot_id,
                listen_port,
                channels,
                sample_rate,
                name,
            } => {
                out.push(1);
                out.extend_from_slice(instance_id);
                out.extend_from_slice(&boot_id.to_le_bytes());
                out.extend_from_slice(&listen_port.to_le_bytes());
                out.extend_from_slice(&channels.to_le_bytes());
                out.extend_from_slice(&sample_rate.to_le_bytes());
                let name = name_bytes(name);
                out.push(name.len() as u8);
                out.extend_from_slice(name);
            }
            Packet::Hello { token } => {
                out.push(2);
                out.extend_from_slice(&token.to_le_bytes());
            }
            Packet::Welcome {
                token,
                channels,
                sample_rate,
            } => {
                out.push(3);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&channels.to_le_bytes());
                out.extend_from_slice(&sample_rate.to_le_bytes());
            }
            Packet::Consumed {
                token,
                stream_frame,
                fill_frames,
                target_frames,
            } => {
                out.push(4);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&stream_frame.to_le_bytes());
                out.extend_from_slice(&fill_frames.to_le_bytes());
                out.extend_from_slice(&target_frames.to_le_bytes());
            }
            Packet::Audio {
                token,
                start_frame,
                channels,
                samples,
            } => {
                out.push(5);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&start_frame.to_le_bytes());
                out.extend_from_slice(&channels.to_le_bytes());
                let frames = samples.len() / (*channels).max(1) as usize;
                out.extend_from_slice(&(frames as u16).to_le_bytes());
                for s in samples {
                    out.extend_from_slice(&s.to_le_bytes());
                }
            }
            Packet::Bye { token } => {
                out.push(6);
                out.extend_from_slice(&token.to_le_bytes());
            }
            Packet::SetName { instance_id, name } => {
                out.push(7);
                out.extend_from_slice(instance_id);
                let name = name_bytes(name);
                out.push(name.len() as u8);
                out.extend_from_slice(name);
            }
            Packet::Reassign {
                instance_id,
                new_id,
            } => {
                out.push(8);
                out.extend_from_slice(instance_id);
                out.extend_from_slice(new_id);
            }
            Packet::Gaps { token, total } => {
                out.push(9);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&total.to_le_bytes());
            }
            Packet::Underruns { token, total } => {
                out.push(10);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&total.to_le_bytes());
            }
            Packet::Nack {
                token,
                start_frame,
                end_frame,
            } => {
                out.push(11);
                out.extend_from_slice(&token.to_le_bytes());
                out.extend_from_slice(&start_frame.to_le_bytes());
                out.extend_from_slice(&end_frame.to_le_bytes());
            }
        }
    }

    /// Returns `None` on wrong magic/version or a malformed body.
    pub fn decode(buf: &[u8]) -> Option<Packet> {
        if buf.len() < HEADER_BYTES || buf[0..4] != MAGIC || buf[4] != VERSION {
            return None;
        }
        let kind = buf[5];
        let mut r = Reader(&buf[HEADER_BYTES..]);
        let packet = match kind {
            1 => {
                let mut instance_id = [0u8; 16];
                instance_id.copy_from_slice(r.take(16)?);
                let boot_id = r.u64()?;
                let listen_port = r.u16()?;
                let channels = r.u16()?;
                let sample_rate = r.u32()?;
                let name_len = r.take(1)?[0] as usize;
                if name_len > MAX_NAME_BYTES {
                    return None;
                }
                let name = String::from_utf8(r.take(name_len)?.to_vec()).ok()?;
                Packet::Beacon {
                    instance_id,
                    boot_id,
                    listen_port,
                    channels,
                    sample_rate,
                    name,
                }
            }
            2 => Packet::Hello { token: r.u64()? },
            3 => Packet::Welcome {
                token: r.u64()?,
                channels: r.u16()?,
                sample_rate: r.u32()?,
            },
            4 => Packet::Consumed {
                token: r.u64()?,
                stream_frame: r.u64()?,
                fill_frames: r.u32()?,
                target_frames: r.u32()?,
            },
            5 => {
                let token = r.u64()?;
                let start_frame = r.u64()?;
                let channels = r.u16()?;
                let frames = r.u16()? as usize;
                let bytes = r.take(frames * channels.max(1) as usize * 4)?;
                let samples = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                Packet::Audio {
                    token,
                    start_frame,
                    channels,
                    samples,
                }
            }
            6 => Packet::Bye { token: r.u64()? },
            7 => {
                let mut instance_id = [0u8; 16];
                instance_id.copy_from_slice(r.take(16)?);
                let name_len = r.take(1)?[0] as usize;
                if name_len > MAX_NAME_BYTES {
                    return None;
                }
                let name = String::from_utf8(r.take(name_len)?.to_vec()).ok()?;
                Packet::SetName { instance_id, name }
            }
            8 => {
                let mut instance_id = [0u8; 16];
                instance_id.copy_from_slice(r.take(16)?);
                let mut new_id = [0u8; 16];
                new_id.copy_from_slice(r.take(16)?);
                Packet::Reassign {
                    instance_id,
                    new_id,
                }
            }
            9 => Packet::Gaps {
                token: r.u64()?,
                total: r.u32()?,
            },
            10 => Packet::Underruns {
                token: r.u64()?,
                total: r.u32()?,
            },
            11 => Packet::Nack {
                token: r.u64()?,
                start_frame: r.u64()?,
                end_frame: r.u64()?,
            },
            _ => return None,
        };
        r.0.is_empty().then_some(packet)
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.0.len() < n {
            return None;
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Some(head)
    }

    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|b| u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Option<u64> {
        self.take(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let packets = [
            Packet::Beacon {
                instance_id: [7; 16],
                boot_id: 0xDEAD_BEEF,
                listen_port: 41000,
                channels: 16,
                sample_rate: 48000,
                name: "LP – Playback A".to_string(),
            },
            Packet::Hello { token: 1 },
            Packet::Welcome {
                token: 2,
                channels: 2,
                sample_rate: 44100,
            },
            Packet::Consumed {
                token: 3,
                stream_frame: u64::MAX / 2,
                fill_frames: 512,
                target_frames: 2048,
            },
            Packet::Audio {
                token: 4,
                start_frame: 123456789,
                channels: 2,
                samples: vec![0.0, 1.0, -1.0, 0.5],
            },
            Packet::Bye { token: 5 },
            Packet::SetName {
                instance_id: [9; 16],
                name: "FOH rack".to_string(),
            },
            Packet::Reassign {
                instance_id: [9; 16],
                new_id: [3; 16],
            },
            Packet::Gaps { token: 6, total: 3 },
            Packet::Underruns { token: 7, total: 2 },
            Packet::Nack {
                token: 8,
                start_frame: 1000,
                end_frame: 1175,
            },
        ];
        let mut buf = Vec::new();
        for p in &packets {
            p.encode(&mut buf);
            assert!(buf.len() <= MAX_PACKET_BYTES);
            assert_eq!(Packet::decode(&buf).as_ref(), Some(p));
        }
    }

    #[test]
    fn audio_fits_mtu_at_16_channels() {
        let frames = frames_per_packet(16);
        assert!(frames >= 1);
        let p = Packet::Audio {
            token: 0,
            start_frame: 0,
            channels: 16,
            samples: vec![0.0; frames * 16],
        };
        let mut buf = Vec::new();
        p.encode(&mut buf);
        assert!(buf.len() <= MAX_PACKET_BYTES);
    }

    #[test]
    fn long_name_truncates_on_char_boundary() {
        // 63 ASCII bytes then a 2-byte char straddling the 64-byte cap: the
        // straddling char must be dropped whole, and the packet must decode.
        let name = format!("{}ří", "x".repeat(63));
        let p = Packet::SetName {
            instance_id: [1; 16],
            name,
        };
        let mut buf = Vec::new();
        p.encode(&mut buf);
        let Some(Packet::SetName { name, .. }) = Packet::decode(&buf) else {
            panic!("truncated name failed to decode");
        };
        assert_eq!(name, "x".repeat(63));
    }

    #[test]
    fn rejects_garbage() {
        assert_eq!(Packet::decode(b"nope"), None);
        assert_eq!(Packet::decode(b"CHBR\x01\x63"), None);
        let mut buf = Vec::new();
        Packet::Hello { token: 9 }.encode(&mut buf);
        buf.push(0);
        assert_eq!(Packet::decode(&buf), None, "trailing bytes rejected");
    }
}
