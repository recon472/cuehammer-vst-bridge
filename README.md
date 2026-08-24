# CueHammer Bridge

A VST3/CLAP plugin that shows up in [CueHammer](https://cuehammer.com) as an
audio output device. Load it in any plugin host — a DAW, LiveProfessor, a
mixing console's plugin rack — and route CueHammer's playback straight into
that host's processing graph, locally or over a wired LAN. Typical use:
getting CueHammer onto an ASIO device another application already owns
(e.g. Dante Virtual Soundcard driven by LiveProfessor on Windows).

## How it works

The plugin has no controls. Insert it, and each instance appears in
CueHammer's Outputs tab as a device under the virtual driver **"VST Bridge"**,
with a generated name like `Amber-Fox-31`. Renaming, buffer leeway, and live
status all live in CueHammer.

The host's audio callback is the clock: the plugin reports what it consumed
over UDP and CueHammer renders exactly that much at the host's sample rate —
no resampling, no drift. Latency is two host audio blocks plus whatever
network leeway you configure in CueHammer (0 for local use). Instance
identity persists in the plugin state chunk, so a saved host session
reconnects to the same CueHammer device on reload.

- Channel layouts: 1 / 2 / 4 / 8 / 16 (one binary; pick per instance)
- Audio passes through, CueHammer's playback is added on top
- Discovery via UDP broadcast beacons on port 27413; audio on an ephemeral
  UDP port per instance
- Wired networks only; DAW offline render/freeze is not supported (the
  bridge delivers real-time audio)

## Building

Requires a Rust toolchain (rustup.rs).

```bash
cargo test                            # protocol + session tests
cargo xtask bundle bridge --release   # produces target/bundled/
```

Install the resulting `CueHammer Bridge.vst3` / `CueHammer Bridge.clap`
into your platform's plugin folder, e.g. on macOS
`~/Library/Audio/Plug-Ins/VST3/` and `~/Library/Audio/Plug-Ins/CLAP/`.

## Repository layout

- `plugin/` — the plugin itself ([nih-plug](https://github.com/robbert-vdh/nih-plug))
- `proto/` — `bridge-proto`, the wire protocol shared with CueHammer
- `xtask/` — bundler (`cargo xtask bundle bridge --release`)

## License

- **`plugin/` is GPL-3.0-or-later** (see [LICENSE](LICENSE)). The VST3
  export is built on GPLv3-licensed VST3 bindings, which makes the plugin a
  GPLv3 work.
- **`proto/` is MIT** (see [proto/LICENSE](proto/LICENSE)), so anyone can
  build compatible senders or endpoints — hardware, other software —
  without inheriting the GPL.

CueHammer itself is a separate, proprietary application; it communicates
with this plugin only over the UDP protocol defined in `proto/`.

VST is a trademark of Steinberg Media Technologies GmbH.
