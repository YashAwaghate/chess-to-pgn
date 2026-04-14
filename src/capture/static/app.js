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

const gridCorrPanel    = document.getElementById('grid-correction-panel');
const gridBoardImg     = document.getElementById('grid-board-img');
const gridCanvas       = document.getElementById('grid-canvas');
const gctx             = gridCanvas.getContext('2d');
const btnConfirmGrid   = document.getElementById('btn-confirm-grid');
const btnRedoOrient    = document.getElementById('btn-redo-orientation');

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
let isCalibrated  = false;
let sessionLoop   = null;
let selectedA1    = null;
let gridLines     = null;
let correctedGrid = null;   // {x_lines: [...], y_lines: [...]}
let gridDragIdx   = -1;     // index of line being dragged (0-8 for x, 9-17 for y)
let gridDragType  = null;   // 'x' or 'y'
const GRID_HIT_R  = 20;    // hit radius in display pixels

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
                save_raw:     document.getElementById('inp-save-raw').checked,
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
            showGridCorrectionStep(data.warped_b64, data.grid);
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

// ─── Step 2: a1 orientation (4-button picker) ───────────────────────────────

const orientBtns = document.querySelectorAll('.orient-btn');

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

    // Draw light grid overlay
    octx.strokeStyle = 'rgba(88, 166, 255, 0.5)';
    octx.lineWidth   = 1;
    xs.forEach(x => {
        octx.beginPath(); octx.moveTo(x * sx, 0); octx.lineTo(x * sx, cH); octx.stroke();
    });
    ys.forEach(y => {
        octx.beginPath(); octx.moveTo(0, y * sy); octx.lineTo(cW, y * sy); octx.stroke();
    });

    // Highlight selected a1 corner
    if (highlighted) {
        const cx0 = xs[highlighted.col]     * sx;
        const cx1 = xs[highlighted.col + 1] * sx;
        const cy0 = ys[highlighted.row]     * sy;
        const cy1 = ys[highlighted.row + 1] * sy;
        octx.fillStyle = 'rgba(46, 160, 67, 0.55)';
        octx.fillRect(cx0, cy0, cx1 - cx0, cy1 - cy0);
        octx.fillStyle    = '#fff';
        octx.font         = `bold ${Math.round(Math.min(cx1 - cx0, cy1 - cy0) * 0.35)}px sans-serif`;
        octx.textAlign    = 'center';
        octx.textBaseline = 'middle';
        octx.fillText('a1', (cx0 + cx1) / 2, (cy0 + cy1) / 2);
    }
}

function selectOrientBtn(col, row) {
    selectedA1 = { col, row };
    btnConfirmA1.disabled = false;
    orientBtns.forEach(b => {
        const bc = parseInt(b.dataset.col);
        const br = parseInt(b.dataset.row);
        b.classList.toggle('selected', bc === col && br === row);
    });
    orientCanvas.width  = warpedImg.naturalWidth  || 400;
    orientCanvas.height = warpedImg.naturalHeight || 400;
    drawOrientationGrid(selectedA1);
    const names = { '0,7': 'Bottom-Left', '7,7': 'Bottom-Right', '0,0': 'Top-Left', '7,0': 'Top-Right' };
    orientHint.innerHTML = `a1 at <strong>${names[`${col},${row}`]}</strong> — confirm?`;
}

orientBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        selectOrientBtn(parseInt(btn.dataset.col), parseInt(btn.dataset.row));
    });
});

function showOrientationStep(warpedB64, grid, suggestedA1) {
    gridLines = grid;
    video.style.display         = 'none';
    overlayCanvas.style.display = 'none';
    calibControls.style.display = 'none';
    orientPanel.style.display   = 'flex';
    warpedImg.src = warpedB64;
    warpedImg.onload = () => {
        orientCanvas.width  = warpedImg.naturalWidth;
        orientCanvas.height = warpedImg.naturalHeight;
        // Mark suggested button
        orientBtns.forEach(b => {
            b.classList.remove('suggested', 'selected');
        });
        if (suggestedA1) {
            orientBtns.forEach(b => {
                if (parseInt(b.dataset.col) === suggestedA1.col &&
                    parseInt(b.dataset.row) === suggestedA1.row) {
                    b.classList.add('suggested');
                }
            });
            // Pre-select the suggestion
            selectOrientBtn(suggestedA1.col, suggestedA1.row);
        } else {
            drawOrientationGrid(null);
        }
    };
    selectedA1             = null;
    btnConfirmA1.disabled  = true;
    btnConfirmA1.innerText = 'Confirm';
    orientHint.innerHTML   = 'Where is <strong>a1</strong>? (White\'s queen-side rook)';
    updateStatusBadge('ORIENTATION');
    // Re-select after onload sets it
    if (suggestedA1) {
        selectedA1 = { col: suggestedA1.col, row: suggestedA1.row };
        btnConfirmA1.disabled = false;
    }
}

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
            infoRotation.innerHTML = `${data.rotation_angle}&deg;`;
            // Grid already confirmed — start session
            isCalibrated = true;
            orientPanel.style.display   = 'none';
            video.style.display         = '';
            overlayCanvas.style.display = '';
            ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
            endGamePanel.style.display  = '';
            startSessionLoop();
        } else {
            alert('Orientation failed: ' + (data.message || ''));
            btnConfirmA1.disabled  = false;
            btnConfirmA1.innerText = 'Confirm';
        }
    } catch (e) {
        console.error(e);
        btnConfirmA1.disabled  = false;
        btnConfirmA1.innerText = 'Confirm';
    }
});

// "Redo Grid" — go back from orientation step to grid correction
btnRedoCorners.addEventListener('click', () => {
    orientPanel.style.display    = 'none';
    gridCorrPanel.style.display  = 'flex';
    selectedA1 = null;
    btnConfirmA1.disabled  = true;
    btnConfirmA1.innerText = 'Confirm';
    orientBtns.forEach(b => b.classList.remove('selected'));
    updateStatusBadge('GRID_CORRECTION');
});

// ─── Step 3: Grid correction ─────────────────────────────────────────────────

function showGridCorrectionStep(warpedB64, grid) {
    correctedGrid = { x_lines: [...grid.x_lines], y_lines: [...grid.y_lines] };
    gridCorrPanel.style.display = 'flex';
    gridBoardImg.src = warpedB64;
    gridBoardImg.onload = () => {
        gridCanvas.width  = gridBoardImg.naturalWidth  || 400;
        gridCanvas.height = gridBoardImg.naturalHeight || 400;
        drawGridLines();
    };
    btnConfirmGrid.disabled  = false;
    btnConfirmGrid.innerText = 'Confirm Grid';
    updateStatusBadge('GRID_CORRECTION');
}

function drawGridLines() {
    if (!correctedGrid) return;
    const cW = gridCanvas.width;
    const cH = gridCanvas.height;
    const iW = gridBoardImg.naturalWidth  || 400;
    const iH = gridBoardImg.naturalHeight || 400;
    const sx = cW / iW;
    const sy = cH / iH;

    gctx.clearRect(0, 0, cW, cH);

    // Draw vertical lines
    correctedGrid.x_lines.forEach((x, i) => {
        const dragging = gridDragType === 'x' && gridDragIdx === i;
        gctx.strokeStyle = dragging ? 'rgba(255,255,100,0.9)' : 'rgba(88, 166, 255, 0.65)';
        gctx.lineWidth   = dragging ? 2.5 : 1.5;
        gctx.beginPath();
        gctx.moveTo(x * sx, 0);
        gctx.lineTo(x * sx, cH);
        gctx.stroke();
    });

    // Draw horizontal lines
    correctedGrid.y_lines.forEach((y, i) => {
        const dragging = gridDragType === 'y' && gridDragIdx === i;
        gctx.strokeStyle = dragging ? 'rgba(255,255,100,0.9)' : 'rgba(88, 166, 255, 0.65)';
        gctx.lineWidth   = dragging ? 2.5 : 1.5;
        gctx.beginPath();
        gctx.moveTo(0, y * sy);
        gctx.lineTo(cW, y * sy);
        gctx.stroke();
    });
}

function hitGridLine(clientX, clientY) {
    const rect = gridCanvas.getBoundingClientRect();
    const dispX = clientX - rect.left;
    const dispY = clientY - rect.top;
    const iW = gridBoardImg.naturalWidth  || 400;
    const iH = gridBoardImg.naturalHeight || 400;
    const sx = rect.width  / iW;
    const sy = rect.height / iH;

    let bestDist = GRID_HIT_R;
    let bestType = null;
    let bestIdx  = -1;

    correctedGrid.x_lines.forEach((x, i) => {
        const d = Math.abs(dispX - x * sx);
        if (d < bestDist) { bestDist = d; bestType = 'x'; bestIdx = i; }
    });
    correctedGrid.y_lines.forEach((y, i) => {
        const d = Math.abs(dispY - y * sy);
        if (d < bestDist) { bestDist = d; bestType = 'y'; bestIdx = i; }
    });

    return bestIdx >= 0 ? { type: bestType, idx: bestIdx } : null;
}

function gridClientToImage(clientX, clientY) {
    const rect = gridCanvas.getBoundingClientRect();
    const iW = gridBoardImg.naturalWidth  || 400;
    const iH = gridBoardImg.naturalHeight || 400;
    return {
        x: (clientX - rect.left) * iW / rect.width,
        y: (clientY - rect.top)  * iH / rect.height,
    };
}

function updateGridDrag(clientX, clientY) {
    if (gridDragIdx < 0 || !gridDragType) return;
    // Outer lines (index 0 and 8) are locked to board edges — not draggable
    if (gridDragIdx === 0 || gridDragIdx === 8) return;
    const img = gridClientToImage(clientX, clientY);
    const lines = gridDragType === 'x' ? correctedGrid.x_lines : correctedGrid.y_lines;
    const val = Math.round(gridDragType === 'x' ? img.x : img.y);
    const maxVal = gridDragType === 'x' ? (gridBoardImg.naturalWidth || 400) : (gridBoardImg.naturalHeight || 400);

    // Constrain: can't cross neighbors, can't go past edges
    const minVal = lines[gridDragIdx - 1] + 1;
    const maxAllowed = lines[gridDragIdx + 1] - 1;
    lines[gridDragIdx] = Math.max(minVal, Math.min(maxAllowed, val));
    drawGridLines();
}

// Mouse handlers
gridCanvas.addEventListener('mousedown', e => {
    const hit = hitGridLine(e.clientX, e.clientY);
    if (hit) { gridDragType = hit.type; gridDragIdx = hit.idx; drawGridLines(); }
});
document.addEventListener('mousemove', e => {
    if (gridDragIdx >= 0) updateGridDrag(e.clientX, e.clientY);
});
document.addEventListener('mouseup', () => {
    if (gridDragIdx >= 0) { gridDragIdx = -1; gridDragType = null; drawGridLines(); }
});

// Touch handlers
gridCanvas.addEventListener('touchstart', e => {
    const t = e.touches[0];
    const hit = hitGridLine(t.clientX, t.clientY);
    if (hit) { gridDragType = hit.type; gridDragIdx = hit.idx; drawGridLines(); e.preventDefault(); }
}, { passive: false });
gridCanvas.addEventListener('touchmove', e => {
    if (gridDragIdx >= 0) { updateGridDrag(e.touches[0].clientX, e.touches[0].clientY); e.preventDefault(); }
}, { passive: false });
gridCanvas.addEventListener('touchend', () => {
    if (gridDragIdx >= 0) { gridDragIdx = -1; gridDragType = null; drawGridLines(); }
});

// Confirm grid
btnConfirmGrid.addEventListener('click', async () => {
    if (!correctedGrid) return;
    btnConfirmGrid.disabled  = true;
    btnConfirmGrid.innerText = 'Confirming…';
    try {
        const res = await fetch('/api/confirm_grid', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x_lines: correctedGrid.x_lines, y_lines: correctedGrid.y_lines }),
        });
        const data = await res.json();
        if (data.status === 'success') {
            gridCorrPanel.style.display = 'none';
            showOrientationStep(data.warped_b64, data.grid, data.suggested_a1);
        } else {
            alert('Grid confirmation failed: ' + (data.message || ''));
            btnConfirmGrid.disabled  = false;
            btnConfirmGrid.innerText = 'Confirm Grid';
        }
    } catch (e) {
        console.error(e);
        btnConfirmGrid.disabled  = false;
        btnConfirmGrid.innerText = 'Confirm Grid';
    }
});

// "Redo Corners" — go back from grid correction to corner calibration
btnRedoOrient.addEventListener('click', () => {
    gridCorrPanel.style.display  = 'none';
    video.style.display          = '';
    overlayCanvas.style.display  = '';
    calibControls.style.display  = '';
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
    selectedA1    = null;
    correctedGrid = null;
    gridCorrPanel.style.display = 'none';
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
        // Reconnect to active session
        setupPanel.style.display   = 'none';
        gamePanel.style.display    = '';
        isCalibrated               = true;
        if (data.game_info) populateGameInfo(data.game_info);
        endGamePanel.style.display = '';
        initCamera().then(startSessionLoop);
    } else if (data.state === 'CALIBRATING' || data.state === 'ORIENTATION' || data.state === 'GRID_CORRECTION') {
        // Stale mid-calibration — reset and start fresh
        fetch('/api/reset', { method: 'POST' });
    }
});
