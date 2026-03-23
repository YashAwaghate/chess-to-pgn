// ─── DOM refs ────────────────────────────────────────────────────────────────
const setupPanel      = document.getElementById('setup-panel');
const gamePanel       = document.getElementById('game-panel');
const btnStartSession = document.getElementById('btn-start-session');

const inpWhite  = document.getElementById('inp-white');
const inpBlack  = document.getElementById('inp-black');
const inpEvent  = document.getElementById('inp-event');
const inpSite   = document.getElementById('inp-site');
const inpDate   = document.getElementById('inp-date');
const inpRound  = document.getElementById('inp-round');
const inpTc     = document.getElementById('inp-tc');
const inpNotes  = document.getElementById('inp-notes');

const video           = document.getElementById('webcam');
const overlayCanvas   = document.getElementById('overlay-canvas');
const ctx             = overlayCanvas.getContext('2d');
const btnCalibrate    = document.getElementById('btn-calibrate');
const btnResetBox     = document.getElementById('btn-reset-box');
const calibControls   = document.getElementById('calibration-controls');

const orientPanel     = document.getElementById('orientation-panel');
const warpedImg       = document.getElementById('warped-board-img');
const orientCanvas    = document.getElementById('orientation-canvas');
const octx            = orientCanvas.getContext('2d');
const btnConfirmA1    = document.getElementById('btn-confirm-a1');
const btnRedoCorners  = document.getElementById('btn-redo-corners');
const orientHint      = document.getElementById('orientation-hint');

const statusBadge    = document.getElementById('app-status');
const infoWhite      = document.getElementById('info-white');
const infoBlack      = document.getElementById('info-black');
const infoEvent      = document.getElementById('info-event');
const infoDate       = document.getElementById('info-date');
const infoTc         = document.getElementById('info-tc');
const infoGameId     = document.getElementById('info-game-id');
const infoMoves      = document.getElementById('info-moves');
const infoRotation   = document.getElementById('info-rotation');
const debugMask      = document.getElementById('debug-mask');
const endGamePanel   = document.getElementById('end-game-panel');
const resultDisplay  = document.getElementById('result-display');
const btnReset       = document.getElementById('btn-reset');

// ─── App state ────────────────────────────────────────────────────────────────
let isCalibrated = false;
let sessionLoop  = null;
let selectedA1   = null;
let gridLines    = null;

// ─── Setup form ──────────────────────────────────────────────────────────────

inpDate.value = new Date().toISOString().slice(0, 10);

function autoFillEvent() {
    const w = inpWhite.value.trim();
    const b = inpBlack.value.trim();
    if (!inpEvent.dataset.userEdited) {
        inpEvent.value = (w || b) ? `${w || 'White'} vs ${b || 'Black'}` : '';
    }
}
inpWhite.addEventListener('input', autoFillEvent);
inpBlack.addEventListener('input', autoFillEvent);
inpEvent.addEventListener('input', () => { inpEvent.dataset.userEdited = '1'; });

btnStartSession.addEventListener('click', async () => {
    const white = inpWhite.value.trim();
    const black = inpBlack.value.trim();
    if (!white || !black) {
        inpWhite.classList.toggle('input-error', !white);
        inpBlack.classList.toggle('input-error', !black);
        return;
    }
    inpWhite.classList.remove('input-error');
    inpBlack.classList.remove('input-error');

    btnStartSession.disabled  = true;
    btnStartSession.innerText = 'Starting…';

    try {
        const res  = await fetch('/api/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                white:        white,
                black:        black,
                event:        inpEvent.value.trim(),
                site:         inpSite.value.trim(),
                game_date:    inpDate.value,
                round:        inpRound.value.trim() || '-',
                time_control: inpTc.value,
                notes:        inpNotes.value.trim(),
            }),
        });
        const data = await res.json();
        if (data.status === 'success') {
            setupPanel.style.display = 'none';
            gamePanel.style.display  = '';
            populateGameInfo(data.game_info);
            updateStatusBadge('CALIBRATING');
            initCamera();
        } else {
            alert('Setup failed: ' + (data.message || ''));
            btnStartSession.disabled  = false;
            btnStartSession.innerText = 'Start Session →';
        }
    } catch (e) {
        console.error(e);
        btnStartSession.disabled  = false;
        btnStartSession.innerText = 'Start Session →';
    }
});

function populateGameInfo(info) {
    infoWhite.innerText = info.white        || '—';
    infoBlack.innerText = info.black        || '—';
    infoEvent.innerText = info.event        || '—';
    infoDate.innerText  = info.date         || '—';
    infoTc.innerText    = info.time_control || '—';
}

// ─── Camera ──────────────────────────────────────────────────────────────────
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width:      { ideal: 1280 },
                height:     { ideal: 960 },
                facingMode: 'environment',
            }
        });
        video.srcObject = stream;
        video.addEventListener('loadedmetadata', () => {
            overlayCanvas.width  = video.videoWidth;
            overlayCanvas.height = video.videoHeight;
            initCropBox();
        }, { once: true });
    } catch (err) {
        console.error(err);
        alert('Could not access camera. Please allow camera permissions and reload.');
    }
}

function getFrameBase64() {
    const tmp = document.createElement('canvas');
    tmp.width  = video.videoWidth;
    tmp.height = video.videoHeight;
    tmp.getContext('2d').drawImage(video, 0, 0);
    return tmp.toDataURL('image/jpeg', 0.85);
}

// ─── Step 1: Crop-box calibration ─────────────────────────────────────────────
//
// cropBox: [TL, TR, BR, BL] in canvas-pixel space.
// Order matches server dst = [[0,0],[400,0],[400,400],[0,400]].
//
// dragIdx 0-3 = corners (TL,TR,BR,BL), 4-7 = edges (top,right,bottom,left).
// Hit detection uses display pixels so the target is consistent at any resolution.

const HIT_R_CORNER = 34;  // display px hit radius for corners
const HIT_R_EDGE   = 28;  // display px hit radius for edge midpoints
let cropBox = null;
let dragIdx = -1;

function defaultCropBox() {
    const w = overlayCanvas.width  || 640;
    const h = overlayCanvas.height || 480;
    // 22% margin each side → box is 56% of frame, corners well inside the view
    const mx = w * 0.22;
    const my = h * 0.22;
    return [
        { x: mx,     y: my },       // 0 TL
        { x: w - mx, y: my },       // 1 TR
        { x: w - mx, y: h - my },   // 2 BR
        { x: mx,     y: h - my },   // 3 BL
    ];
}

function initCropBox() {
    cropBox = defaultCropBox();
    drawCropBox();
}

// Returns canvas→display scale + bounding rect (one getBoundingClientRect call).
function getCanvasScale() {
    const rect = overlayCanvas.getBoundingClientRect();
    return {
        x: rect.width  > 0 ? overlayCanvas.width  / rect.width  : 1,
        y: rect.height > 0 ? overlayCanvas.height / rect.height : 1,
        rect,
    };
}

// Edge midpoints derived from current cropBox corners.
// Index offset +4: top=4, right=5, bottom=6, left=7
function edgeMidpoints() {
    const [tl, tr, br, bl] = cropBox;
    return [
        { x: (tl.x + tr.x) / 2, y: (tl.y + tr.y) / 2 },  // 4 top
        { x: (tr.x + br.x) / 2, y: (tr.y + br.y) / 2 },  // 5 right
        { x: (br.x + bl.x) / 2, y: (br.y + bl.y) / 2 },  // 6 bottom
        { x: (bl.x + tl.x) / 2, y: (bl.y + tl.y) / 2 },  // 7 left
    ];
}

function drawCropBox() {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (!cropBox) return;

    const { x: sx, y: sy } = getCanvasScale();
    // Handle draw radii in canvas pixels (≈ target display px)
    const cornerR = 13 * Math.max(sx, sy);
    const edgeR   =  9 * Math.max(sx, sy);

    // ── Quad fill + outline ───────────────────────────────────────────────────
    ctx.beginPath();
    ctx.moveTo(cropBox[0].x, cropBox[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(cropBox[i].x, cropBox[i].y);
    ctx.closePath();
    ctx.fillStyle   = 'rgba(88, 166, 255, 0.12)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(88, 166, 255, 0.85)';
    ctx.lineWidth   = Math.max(1.5, 2 * Math.max(sx, sy));
    ctx.stroke();

    // ── Edge midpoint handles (drawn first so corners appear on top) ──────────
    const edges = edgeMidpoints();
    edges.forEach((p, i) => {
        const active = (dragIdx === i + 4);
        ctx.beginPath();
        ctx.arc(p.x, p.y, edgeR, 0, 2 * Math.PI);
        ctx.fillStyle   = active ? '#ffffff' : 'rgba(88, 166, 255, 0.65)';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth   = Math.max(1, 1.5 * Math.max(sx, sy));
        ctx.stroke();
    });

    // ── Corner handles ────────────────────────────────────────────────────────
    const labels = ['TL', 'TR', 'BR', 'BL'];
    cropBox.forEach((p, i) => {
        const active = (dragIdx === i);
        ctx.beginPath();
        ctx.arc(p.x, p.y, cornerR, 0, 2 * Math.PI);
        ctx.fillStyle   = active ? '#ffffff' : 'rgba(88, 166, 255, 0.92)';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth   = Math.max(1.5, 2 * Math.max(sx, sy));
        ctx.stroke();

        ctx.fillStyle       = active ? '#1f6feb' : '#ffffff';
        ctx.font            = `bold ${Math.round(cornerR * 0.65)}px sans-serif`;
        ctx.textAlign       = 'center';
        ctx.textBaseline    = 'middle';
        ctx.fillText(labels[i], p.x, p.y);
    });

    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
}

// Hit test all 8 handles (corners 0-3, edges 4-7) in display pixel space.
function hitHandleAt(clientX, clientY) {
    if (!cropBox) return -1;
    const { x: sx, y: sy, rect } = getCanvasScale();

    const allHandles = [
        ...cropBox,          // 0-3 corners
        ...edgeMidpoints(),  // 4-7 edges
    ];
    const hitR = [
        HIT_R_CORNER, HIT_R_CORNER, HIT_R_CORNER, HIT_R_CORNER,
        HIT_R_EDGE,   HIT_R_EDGE,   HIT_R_EDGE,   HIT_R_EDGE,
    ];

    let best = -1, bestD = Infinity;
    allHandles.forEach((p, i) => {
        const dispX = p.x / sx + rect.left;
        const dispY = p.y / sy + rect.top;
        const d = Math.hypot(clientX - dispX, clientY - dispY);
        if (d < hitR[i] && d < bestD) { best = i; bestD = d; }
    });
    return best;
}

// Client (display) coords → canvas pixel coords, clamped to canvas bounds.
function clientToCanvas(clientX, clientY) {
    const { x: sx, y: sy, rect } = getCanvasScale();
    return {
        x: Math.max(0, Math.min(overlayCanvas.width,  (clientX - rect.left) * sx)),
        y: Math.max(0, Math.min(overlayCanvas.height, (clientY - rect.top)  * sy)),
    };
}

function startDrag(idx) {
    dragIdx = idx;
    drawCropBox();
}

function moveDrag(clientX, clientY) {
    if (dragIdx < 0 || isCalibrated) return;
    const pt = clientToCanvas(clientX, clientY);
    if (dragIdx < 4) {
        // Corner — free movement
        cropBox[dragIdx] = pt;
    } else {
        // Edge — constrained to one axis
        switch (dragIdx) {
            case 4: cropBox[0].y = cropBox[1].y = pt.y; break;  // top   → Y only
            case 5: cropBox[1].x = cropBox[2].x = pt.x; break;  // right → X only
            case 6: cropBox[2].y = cropBox[3].y = pt.y; break;  // bottom→ Y only
            case 7: cropBox[0].x = cropBox[3].x = pt.x; break;  // left  → X only
        }
    }
    drawCropBox();
}

function stopDrag() {
    if (dragIdx >= 0) { dragIdx = -1; drawCropBox(); }
}

// ── Mouse (laptop / desktop) ─────────────────────────────────────────────────
overlayCanvas.addEventListener('mousedown', e => {
    if (isCalibrated) return;
    const idx = hitHandleAt(e.clientX, e.clientY);
    if (idx >= 0) {
        startDrag(idx);
        e.preventDefault();
    }
});

// Move and up on document so the drag continues even if the mouse leaves the canvas.
document.addEventListener('mousemove', e => {
    if (dragIdx < 0 || isCalibrated) return;
    moveDrag(e.clientX, e.clientY);
});

document.addEventListener('mouseup', stopDrag);

// ── Touch (mobile) ───────────────────────────────────────────────────────────
overlayCanvas.addEventListener('touchstart', e => {
    if (isCalibrated) return;
    const t = e.touches[0];
    const idx = hitHandleAt(t.clientX, t.clientY);
    if (idx >= 0) {
        startDrag(idx);
        e.preventDefault();   // prevent scroll during drag
    }
}, { passive: false });

overlayCanvas.addEventListener('touchmove', e => {
    if (dragIdx < 0 || isCalibrated) return;
    const t = e.touches[0];
    moveDrag(t.clientX, t.clientY);
    e.preventDefault();
}, { passive: false });

overlayCanvas.addEventListener('touchend',    stopDrag);
overlayCanvas.addEventListener('touchcancel', stopDrag);

// ── Controls ─────────────────────────────────────────────────────────────────
btnResetBox.addEventListener('click', () => {
    cropBox = defaultCropBox();
    drawCropBox();
});

btnCalibrate.addEventListener('click', async () => {
    if (!cropBox) return;
    btnCalibrate.disabled  = true;
    btnCalibrate.innerText = 'Calibrating…';
    try {
        const res  = await fetch('/api/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // cropBox order [TL,TR,BR,BL] matches server dst [[0,0],[400,0],[400,400],[0,400]]
            body: JSON.stringify({ points: cropBox, image_b64: getFrameBase64() }),
        });
        const data = await res.json();
        if (data.status === 'success') {
            showOrientationStep(data.warped_b64, data.grid);
        } else {
            alert('Calibration failed: ' + (data.message || ''));
            btnCalibrate.innerText = 'Calibrate';
            btnCalibrate.disabled  = false;
        }
    } catch (e) {
        console.error(e);
        btnCalibrate.innerText = 'Calibrate';
        btnCalibrate.disabled  = false;
    }
});

// ─── Step 2: a1 orientation ──────────────────────────────────────────────────

function pixelToCell(imgX, imgY) {
    const xs = gridLines.x_lines;
    const ys = gridLines.y_lines;
    let col = 0, row = 0;
    for (let i = 0; i < 8; i++) { if (imgX >= xs[i]) col = i; }
    for (let i = 0; i < 8; i++) { if (imgY >= ys[i]) row = i; }
    return { col, row };
}

function snapToCorner(col, row) {
    const corners = [{ col: 0, row: 0 }, { col: 7, row: 0 }, { col: 0, row: 7 }, { col: 7, row: 7 }];
    return corners.reduce((best, c) => {
        const d  = Math.abs(c.col - col) + Math.abs(c.row - row);
        const bd = Math.abs(best.col - col) + Math.abs(best.row - row);
        return d < bd ? c : best;
    });
}

function drawOrientationGrid(highlighted) {
    const xs = gridLines ? gridLines.x_lines : [...Array(9)].map((_, i) => i * 50);
    const ys = gridLines ? gridLines.y_lines : [...Array(9)].map((_, i) => i * 50);
    const iW = warpedImg.naturalWidth  || 400;
    const iH = warpedImg.naturalHeight || 400;
    const cW = orientCanvas.width;
    const cH = orientCanvas.height;
    const sx = cW / iW;
    const sy = cH / iH;

    octx.clearRect(0, 0, cW, cH);

    if (highlighted) {
        const cx0 = xs[highlighted.col]     * sx;
        const cx1 = xs[highlighted.col + 1] * sx;
        const cy0 = ys[highlighted.row]     * sy;
        const cy1 = ys[highlighted.row + 1] * sy;
        octx.fillStyle = 'rgba(46, 160, 67, 0.55)';
        octx.fillRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
        octx.fillStyle    = '#fff';
        octx.font         = `bold ${Math.round(Math.min(cx1 - cx0, cy1 - cy0) * 0.3)}px sans-serif`;
        octx.textAlign    = 'center';
        octx.textBaseline = 'middle';
        octx.fillText('a1', (cx0 + cx1) / 2, (cy0 + cy1) / 2);
    }

    octx.strokeStyle = 'rgba(88, 166, 255, 0.7)';
    octx.lineWidth   = 1.5;
    xs.forEach(x => {
        octx.beginPath(); octx.moveTo(x * sx, 0); octx.lineTo(x * sx, cH); octx.stroke();
    });
    ys.forEach(y => {
        octx.beginPath(); octx.moveTo(0, y * sy); octx.lineTo(cW, y * sy); octx.stroke();
    });

    const cornerLabels = [
        { col: 0, row: 0, text: 'TL' }, { col: 7, row: 0, text: 'TR' },
        { col: 0, row: 7, text: 'BL' }, { col: 7, row: 7, text: 'BR' },
    ];
    const labelSize = Math.round(Math.min(cW, cH) / 8 * 0.26);
    octx.fillStyle    = 'rgba(255,255,255,0.45)';
    octx.font         = `${labelSize}px sans-serif`;
    octx.textAlign    = 'center';
    octx.textBaseline = 'middle';
    cornerLabels.forEach(l => {
        if (highlighted && l.col === highlighted.col && l.row === highlighted.row) return;
        const mx = (xs[l.col] + xs[l.col + 1]) / 2 * sx;
        const my = (ys[l.row] + ys[l.row + 1]) / 2 * sy;
        octx.fillText(l.text, mx, my);
    });
    octx.textAlign    = 'left';
    octx.textBaseline = 'alphabetic';
}

function showOrientationStep(warpedB64, grid) {
    gridLines = grid;
    video.style.display         = 'none';
    overlayCanvas.style.display = 'none';
    calibControls.style.display = 'none';
    orientPanel.style.display   = 'block';
    warpedImg.src = warpedB64;
    warpedImg.onload = () => {
        orientCanvas.width  = warpedImg.naturalWidth;
        orientCanvas.height = warpedImg.naturalHeight;
        drawOrientationGrid(null);
    };
    selectedA1             = null;
    btnConfirmA1.disabled  = true;
    btnConfirmA1.innerText = 'Confirm a1';
    orientHint.innerHTML   = 'Tap the <strong>a1</strong> square (bottom-left from White\'s view)';
    updateStatusBadge('ORIENTATION');
}

function handleOrientSelect(clientX, clientY) {
    const rect = orientCanvas.getBoundingClientRect();
    const imgX = (clientX - rect.left) * (warpedImg.naturalWidth  / rect.width);
    const imgY = (clientY - rect.top)  * (warpedImg.naturalHeight / rect.height);

    const cell = pixelToCell(imgX, imgY);
    selectedA1 = snapToCorner(cell.col, cell.row);
    btnConfirmA1.disabled = false;

    orientCanvas.width  = warpedImg.naturalWidth;
    orientCanvas.height = warpedImg.naturalHeight;
    drawOrientationGrid(selectedA1);

    const names = { '0,7': 'Bottom-Left', '7,7': 'Bottom-Right', '0,0': 'Top-Left', '7,0': 'Top-Right' };
    orientHint.innerHTML = `a1 at <strong>${names[`${selectedA1.col},${selectedA1.row}`]}</strong> — confirm?`;
}

orientCanvas.addEventListener('mousedown', e => {
    handleOrientSelect(e.clientX, e.clientY);
});
orientCanvas.addEventListener('touchstart', e => {
    const t = e.touches[0];
    handleOrientSelect(t.clientX, t.clientY);
    e.preventDefault();
}, { passive: false });

btnConfirmA1.addEventListener('click', async () => {
    if (!selectedA1) return;
    btnConfirmA1.disabled  = true;
    btnConfirmA1.innerText = 'Confirming…';
    try {
        const res  = await fetch('/api/set_orientation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ col: selectedA1.col, row: selectedA1.row }),
        });
        const data = await res.json();
        if (data.status === 'success') {
            isCalibrated = true;
            orientPanel.style.display   = 'none';
            video.style.display         = '';
            overlayCanvas.style.display = '';
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            infoRotation.innerHTML      = `${data.rotation_angle}&deg;`;
            endGamePanel.style.display  = '';
            startSessionLoop();
        } else {
            alert('Orientation failed: ' + (data.message || ''));
            btnConfirmA1.disabled  = false;
            btnConfirmA1.innerText = 'Confirm a1';
        }
    } catch (e) {
        console.error(e);
        btnConfirmA1.disabled  = false;
        btnConfirmA1.innerText = 'Confirm a1';
    }
});

btnRedoCorners.addEventListener('click', () => {
    orientPanel.style.display   = 'none';
    video.style.display         = '';
    overlayCanvas.style.display = '';
    calibControls.style.display = '';
    selectedA1 = null;
    btnCalibrate.disabled  = false;
    btnCalibrate.innerText = 'Calibrate';
    cropBox = defaultCropBox();
    drawCropBox();
    updateStatusBadge('CALIBRATING');
});

// ─── End game / result ────────────────────────────────────────────────────────
document.querySelectorAll('.result-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
        const result = btn.dataset.result;
        try {
            const res  = await fetch('/api/end_game', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ result }),
            });
            const data = await res.json();
            if (data.status === 'success') {
                resultDisplay.innerText     = `Result recorded: ${result}`;
                resultDisplay.style.display = '';
                document.querySelectorAll('.result-btn').forEach(b => b.disabled = true);
            }
        } catch (e) { console.error(e); }
    });
});

// ─── Session loop ─────────────────────────────────────────────────────────────
function updateStatusBadge(state) {
    statusBadge.innerText = state;
    statusBadge.className = 'status-badge ' + state.toLowerCase();
}

function updateUI(data) {
    updateStatusBadge(data.state);
    infoGameId.innerText = data.game_id    || '—';
    infoMoves.innerText  = data.move_number > 0 ? data.move_number - 1 : 0;
    if (data.rotation_angle !== undefined)
        infoRotation.innerHTML = `${data.rotation_angle}&deg;`;
    if (data.game_info) populateGameInfo(data.game_info);
}

async function fetchState() {
    const res  = await fetch('/api/state');
    const data = await res.json();
    updateUI(data);
    return data;
}

function startSessionLoop() {
    if (sessionLoop) clearInterval(sessionLoop);
    fetchState();
    sessionLoop = setInterval(async () => {
        if (!isCalibrated) return;
        try {
            const res  = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_b64: getFrameBase64() }),
            });
            const data = await res.json();
            if (data.state)    updateStatusBadge(data.state);
            if (data.mask_b64) debugMask.src = data.mask_b64;
            if (Math.random() < 0.2) fetchState();
        } catch (e) { console.error('Frame drop:', e); }
    }, 200);
}

btnReset.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    if (sessionLoop) clearInterval(sessionLoop);
    isCalibrated = false;
    cropBox      = null;
    selectedA1   = null;
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    gamePanel.style.display     = 'none';
    setupPanel.style.display    = '';
    endGamePanel.style.display  = 'none';
    resultDisplay.style.display = 'none';
    document.querySelectorAll('.result-btn').forEach(b => b.disabled = false);
    btnCalibrate.disabled  = false;
    btnCalibrate.innerText = 'Calibrate';
    btnStartSession.disabled  = false;
    btnStartSession.innerText = 'Start Session →';
    delete inpEvent.dataset.userEdited;
    updateStatusBadge('SETUP');
});

// ─── Boot ─────────────────────────────────────────────────────────────────────
fetchState().then(data => {
    if (data.state === 'STATIC' || data.state === 'MOVING') {
        setupPanel.style.display   = 'none';
        gamePanel.style.display    = '';
        isCalibrated               = true;
        if (data.game_info) populateGameInfo(data.game_info);
        endGamePanel.style.display = '';
        initCamera().then(startSessionLoop);
    }
});
