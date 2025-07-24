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
