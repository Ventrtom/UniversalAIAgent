"""JavaScript helpers for the Gradio interface."""

AUTO_STOP_START_JS = """
() => {
    const el = document.getElementById('mic_recorder');
    if (!el) return [];
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const ctx = new AudioContext();
    navigator.mediaDevices.getUserMedia({audio:true}).then(stream => {
        window.__autoStop_stream = stream;
        const src = ctx.createMediaStreamSource(stream);
        const proc = ctx.createScriptProcessor(1024,1,1);
        window.__autoStop_ctx = ctx;
        window.__autoStop_proc = proc;
        let silenceStart = ctx.currentTime;
        proc.onaudioprocess = e => {
            const buf = e.inputBuffer.getChannelData(0);
            let max = 0;
            for (let i=0;i<buf.length;i++) if (Math.abs(buf[i])>max) max=Math.abs(buf[i]);
            if (max>0.015) {
                silenceStart = ctx.currentTime;
            } else if (ctx.currentTime - silenceStart > 5) {
                const stopBtn = el.querySelector('button[aria-label*="Stop"]');
                if (stopBtn) stopBtn.click();
            }
        };
        src.connect(proc);
        proc.connect(ctx.destination);
    });
    return [];
}
"""

AUTO_STOP_STOP_JS = """
() => {
    if (window.__autoStop_proc) window.__autoStop_proc.disconnect();
    if (window.__autoStop_ctx) window.__autoStop_ctx.close();
    if (window.__autoStop_stream) window.__autoStop_stream.getTracks().forEach(t=>t.stop());
    window.__autoStop_proc = null;
    window.__autoStop_ctx = null;
    window.__autoStop_stream = null;
    return [];
}
"""

MIC_TOGGLE_RECORD_JS = """
() => {
    const el = document.getElementById('mic_recorder');
    if (!el) return [];
    // Pokud už nahráváme -> zastav
    const stopBtn = el.querySelector('button[aria-label*="Stop"]');
    if (stopBtn) { stopBtn.click(); return []; }
    // Jinak spusť nahrávání
    const recordBtn = el.querySelector('button[aria-label*="Record"], button[aria-label*="Start"]');
    if (recordBtn) recordBtn.click();
    return [];
}
"""


START_ONE_SHOT_RECORD_JS = """
() => {
  const constraints = { audio: { echoCancellation: true, noiseSuppression: true } };
  const SILENCE_MS = 1200;      // stop after this long below threshold
  const MIN_MS = 800;           // don't stop too early
  const THRESHOLD = 0.02;       // simple RMS gate

  return navigator.mediaDevices.getUserMedia(constraints).then(stream => {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const source = ctx.createMediaStreamSource(stream);
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    let chunks = [];
    let silenceStart = null;
    let startedAt = performance.now();

    // Prefer webm for wide support
    const __mime = (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported('audio/webm')) ? 'audio/webm' : (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus') ? 'audio/ogg;codecs=opus' : '');
    const rec = __mime ? new MediaRecorder(stream, { mimeType: __mime }) : new MediaRecorder(stream);
    rec.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };

    processor.onaudioprocess = e => {
      const input = e.inputBuffer.getChannelData(0);
      let rms = 0;
      for (let i = 0; i < input.length; i++) rms += input[i] * input[i];
      rms = Math.sqrt(rms / input.length);

      const now = performance.now();
      if (rms < THRESHOLD) {
        if (silenceStart === null) silenceStart = now;
        if ((now - silenceStart) >= SILENCE_MS && (now - startedAt) >= MIN_MS) {
          try { processor.disconnect(); } catch {}
          try { source.disconnect(); } catch {}
          try { ctx.close(); } catch {}
          rec.stop();
          stream.getTracks().forEach(t => t.stop());
        }
      } else {
        silenceStart = null;
      }
    };

    source.connect(processor);
    processor.connect(ctx.destination);
    rec.start();

    return new Promise(resolve => {
      rec.onstop = () => {
        const blob = new Blob(chunks, { type: __mime || 'audio/webm' });
      /* Gradio (filepath mode) vyžaduje File objekt s názvem */
      const file = new File(
        [blob],
        `recording_${Date.now()}.webm`,
        { type: __mime || 'audio/webm' },
      );
      /* Markdown update musí být dict => {value, visible} */
      resolve([
        file,
        { value: "⏳ Přepis…", visible: true },
      ]);
      };
    });
  });
}
"""


TOGGLE_WAKE_LISTENER_JS = """
(enabled) => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    return ["❗ Prohlížeč nepodporuje SpeechRecognition (zkuste Chrome)."];
  }
  if (!window.__wake) window.__wake = {};
  if (!enabled) {
    window.__wake.enabled = false;
    if (window.__wake.recog) { try { window.__wake.recog.stop(); } catch {} }
    window.__wake.recog = null;
    return ["🛑 Auto‑listening vypnuto."];
  }
  // start
  const recog = new SR();
  recog.continuous = true;
  recog.interimResults = true;
  recog.lang = "en-US"; // wake word: "hey agent"
  window.__wake.enabled = true;
  window.__wake.recog = recog;

  let lastTriggerAt = 0;
  const REARM_MS = 3000;

  recog.onresult = (e) => {
    let txt = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      txt += e.results[i][0].transcript.toLowerCase();
    }
    if (txt.includes("hey agent")) {
      const now = Date.now();
      if (now - lastTriggerAt > REARM_MS) {
        lastTriggerAt = now;
        // Simulate click on mic button to reuse its one-shot record + pipeline
        const btn = document.getElementById("mic_button");
        if (btn) btn.click();
      }
    }
  };
  recog.onend = () => { if (window.__wake.enabled) { try { recog.start(); } catch {} } };

  try { recog.start(); } catch {}
  return ["👂 Auto‑listening zapnuto (čekám na 'hey agent'…)."];
}
"""
