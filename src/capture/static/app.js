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
const btnAutoCorners  = document.getElementById('btn-auto-corners');
const calibControls   = document.getElementById('calibration-controls');
const btnCalMenuToggle = document.getElementById('btn-cal-menu-toggle');

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
const debugMask          = document.getElementById('debug-mask');
const endGamePanel       = document.getElementById('end-game-panel');
const resultDisplay      = document.getElementById('result-display');
const btnReset           = document.getElementById('btn-reset');
const boardPreviewPanel  = document.getElementById('board-preview-panel');
const boardPreviewImg    = document.getElementById('board-preview-img');
const boardPreviewMove   = document.getElementById('board-preview-move');

// ─── App state ────────────────────────────────────────────────────────────────
let isCalibrated  = false;
let sessionLoop   = null;
let selectedA1    = null;
let gridLines     = null;
let correctedGrid = null;   // {x_lines: [...], y_lines: [...]}
let gridDragIdx   = -1;     // index of line being dragged (0-8 for x, 9-17 for y)
let gridDragType  = null;   // 'x' or 'y'
const GRID_HIT_R  = 20;    // hit radius in display pixels

function setCalibrationMenuOpen(open) {
    if (!calibControls || !btnCalMenuToggle) return;
    calibControls.classList.toggle('open', open);
    btnCalMenuToggle.classList.toggle('open', open);
    btnCalMenuToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
}

// ─── Setup form ──────────────────────────────────────────────────────────────

// Set today's date on hidden field
document.getElementById('inp-date').value = new Date().toISOString().slice(0, 10);

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
                save_raw:     false,
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
    if (!video.videoWidth || !video.videoHeight) return null;
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
let autoDetectHighlightUntil = 0;

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
    const autoDetected = performance.now() < autoDetectHighlightUntil;

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
        ctx.fillStyle   = active ? '#ffffff' : (autoDetected ? 'rgba(46, 160, 67, 0.75)' : 'rgba(88, 166, 255, 0.65)');
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
        ctx.fillStyle   = active ? '#ffffff' : (autoDetected ? 'rgba(46, 160, 67, 0.95)' : 'rgba(88, 166, 255, 0.92)');
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

function detectedPointsToCanvas(points, sourceWidth, sourceHeight) {
    const srcW = sourceWidth  || video.videoWidth  || overlayCanvas.width;
    const srcH = sourceHeight || video.videoHeight || overlayCanvas.height;
    const dstW = overlayCanvas.width  || srcW;
    const dstH = overlayCanvas.height || srcH;
    const normalized = points.every(p =>
        Number.isFinite(p.x) && Number.isFinite(p.y) &&
        p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1
    );

    return points.map(p => {
        const srcX = normalized ? p.x * srcW : p.x;
        const srcY = normalized ? p.y * srcH : p.y;
        return {
            x: Math.max(0, Math.min(dstW, srcX * dstW / srcW)),
            y: Math.max(0, Math.min(dstH, srcY * dstH / srcH)),
        };
    });
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
    setCalibrationMenuOpen(false);
    cropBox = defaultCropBox();
    drawCropBox();
});

btnCalMenuToggle.addEventListener('click', () => {
    setCalibrationMenuOpen(!calibControls.classList.contains('open'));
});

btnAutoCorners.addEventListener('click', async () => {
    setCalibrationMenuOpen(false);
    const frameB64 = getFrameBase64();
    if (!frameB64) return;
    const orig = btnAutoCorners.innerText;
    btnAutoCorners.disabled  = true;
    btnAutoCorners.innerText = 'Detecting…';
    try {
        const res  = await fetch('/api/detect_corners', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ image_b64: frameB64 }),
        });
        const data = await res.json();
        if (data.status === 'success' && data.points && data.points.length === 4) {
            // Server returns [TL, TR, BR, BL] in source-frame pixel space.
            cropBox = detectedPointsToCanvas(data.points, data.image_width, data.image_height);
            autoDetectHighlightUntil = performance.now() + 2000;
            drawCropBox();
            setTimeout(drawCropBox, 2100);
            btnAutoCorners.innerText = 'Auto-detect ✓';
            setTimeout(() => {
                btnAutoCorners.innerText = orig;
                btnAutoCorners.disabled  = false;
            }, 2000);
        } else {
            alert('Auto-detect failed: ' + (data.message || 'no corners found'));
            btnAutoCorners.innerText = orig;
            btnAutoCorners.disabled  = false;
        }
    } catch (e) {
        console.error(e);
        btnAutoCorners.innerText = orig;
        btnAutoCorners.disabled  = false;
    }
});

btnCalibrate.addEventListener('click', async () => {
    setCalibrationMenuOpen(false);
    if (!cropBox) return;
    const frameB64 = getFrameBase64();
    if (!frameB64) {
        alert('Camera not ready yet. Please wait a moment and try again.');
        return;
    }
    btnCalibrate.disabled  = true;
    btnCalibrate.innerText = 'Calibrating…';
    try {
        const res  = await fetch('/api/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // cropBox order [TL,TR,BR,BL] matches server dst [[0,0],[400,0],[400,400],[0,400]]
            body: JSON.stringify({ points: cropBox, image_b64: frameB64 }),
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
    const img = gridClientToImage(clientX, clientY);
    const lines = gridDragType === 'x' ? correctedGrid.x_lines : correctedGrid.y_lines;
    const val = Math.round(gridDragType === 'x' ? img.x : img.y);
    const maxVal = gridDragType === 'x' ? (gridBoardImg.naturalWidth || 400) : (gridBoardImg.naturalHeight || 400);

    // Constrain: can't cross neighbors, can't go past image edges
    const minVal = gridDragIdx === 0 ? 0 : lines[gridDragIdx - 1] + 1;
    const maxAllowed = gridDragIdx === 8 ? maxVal : lines[gridDragIdx + 1] - 1;
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
const pgnScreen      = document.getElementById('pgn-screen');
const pgnText        = document.getElementById('pgn-text');
const pgnMeta        = document.getElementById('pgn-meta');
const pgnBoard       = document.getElementById('pgn-board');
const pgnMoveList    = document.getElementById('pgn-move-list');
const pgnFenDisplay  = document.getElementById('pgn-fen-display');
const pgnMoveLabel   = document.getElementById('pgn-move-label');
const btnCopyPgn     = document.getElementById('btn-copy-pgn');
const btnDownloadPgn = document.getElementById('btn-download-pgn');
const btnNewSameCal  = document.getElementById('btn-new-same-cal');
const btnNewFreshCal = document.getElementById('btn-new-fresh-cal');
const btnGoStart     = document.getElementById('btn-go-start');
const btnGoEnd       = document.getElementById('btn-go-end');
const btnGoPrev      = document.getElementById('btn-go-prev');
const btnGoNext      = document.getElementById('btn-go-next');

let lastGameId      = null;
let pgnFenSequence  = [];   // one FEN per frame (index 0 = starting position)
let pgnMoves        = [];   // list of SAN strings
let pgnMoveTags     = [];   // 'sure'|'prior'|'failed' per move
let pgnMoveConfs    = [];   // confidence float per move
let currentMoveIdx  = -1;  // -1 = start position, 0..N-1 = after move N

const PIECE_SYMBOLS = {
    P:'♙', N:'♘', B:'♗', R:'♖', Q:'♕', K:'♔',
    p:'♟', n:'♞', b:'♝', r:'♜', q:'♛', k:'♚',
};

function parseFen(fen) {
    const board = [];
    const rows = fen.split(' ')[0].split('/');
    for (const row of rows) {
        const cells = [];
        for (const ch of row) {
            if (ch >= '1' && ch <= '8') {
                for (let i = 0; i < parseInt(ch); i++) cells.push('');
            } else {
                cells.push(ch);
            }
        }
        board.push(cells);
    }
    return board;
}

function renderBoard(fen, highlightSqs) {
    if (!pgnBoard) return;
    pgnBoard.innerHTML = '';
    const board = parseFen(fen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    const hl = new Set(highlightSqs || []);
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement('div');
            const isLight = (r + c) % 2 === 0;
            cell.className = 'board-cell ' + (isLight ? 'light' : 'dark');
            const piece = board[r] && board[r][c];
            if (piece) cell.textContent = PIECE_SYMBOLS[piece] || '';
            // Highlight squares by algebraic name (e.g. "e4")
            const file = String.fromCharCode(97 + c);
            const rank = 8 - r;
            if (hl.has(file + rank)) cell.classList.add('highlight');
            pgnBoard.appendChild(cell);
        }
    }
    if (pgnFenDisplay) pgnFenDisplay.value = fen || '';
}

function renderMoveList(moves, tags, confs) {
    if (!pgnMoveList) return;
    pgnMoveList.innerHTML = '';
    for (let i = 0; i < moves.length; i++) {
        const isWhite = i % 2 === 0;
        const item = document.createElement('div');
        item.className = 'move-item';
        item.dataset.idx = i;

        const num = document.createElement('span');
        num.className = 'move-num';
        if (isWhite) num.textContent = Math.floor(i / 2 + 1) + '.';
        item.appendChild(num);

        const san = document.createElement('span');
        san.className = 'move-san';
        san.textContent = moves[i] || '?';
        item.appendChild(san);

        const tag = (tags && tags[i]) || 'sure';
        const conf = confs && confs[i] != null ? confs[i] : null;
        const badge = document.createElement('span');
        badge.className = `conf-badge ${tag}`;
        badge.textContent = tag === 'sure' ? '✓' : tag === 'prior' ? '~' : '✗';
        if (conf !== null) badge.title = `${Math.round(conf * 100)}% conf`;
        item.appendChild(badge);

        item.addEventListener('click', () => gotoMove(i));
        pgnMoveList.appendChild(item);
    }
}

function gotoMove(idx) {
    currentMoveIdx = idx;
    // Use the FEN sequence: index 0 = before any moves, move i produces fen at i+1
    const fenIdx = idx + 1;
    const fen = pgnFenSequence[fenIdx] || pgnFenSequence[pgnFenSequence.length - 1] || '';
    renderBoard(fen);

    const moveNum = Math.floor(idx / 2) + 1;
    const color   = idx % 2 === 0 ? 'W' : 'B';
    pgnMoveLabel.textContent = `Move ${moveNum}${color}`;

    // Highlight active item in move list
    pgnMoveList.querySelectorAll('.move-item').forEach(el => {
        el.classList.toggle('active', parseInt(el.dataset.idx) === idx);
    });
    // Scroll active move into view
    const activeEl = pgnMoveList.querySelector('.move-item.active');
    if (activeEl) activeEl.scrollIntoView({ block: 'nearest' });
}

function gotoStart() {
    currentMoveIdx = -1;
    const fen = pgnFenSequence[0] || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    renderBoard(fen);
    pgnMoveLabel.textContent = 'Start';
    pgnMoveList.querySelectorAll('.move-item').forEach(el => el.classList.remove('active'));
}

async function showPgnScreen(result) {
    pgnScreen.style.display = 'flex';
    pgnText.value     = 'Generating PGN…';
    pgnMeta.innerHTML = '';
    pgnBoard.innerHTML   = '';
    pgnMoveList.innerHTML = '';
    pgnFenSequence = [];
    pgnMoves       = [];
    pgnMoveTags    = [];
    pgnMoveConfs   = [];
    currentMoveIdx = -1;

    try {
        const stateRes  = await fetch('/api/state');
        const stateData = await stateRes.json();
        lastGameId = stateData.game_id;

        const res  = await fetch(`/api/generate_pgn/${lastGameId}`, { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            pgnText.value = data.pgn || '(empty)';

            // Store enriched data
            pgnMoves      = data.moves       || [];
            pgnMoveTags   = data.move_tags   || [];
            pgnMoveConfs  = data.move_confidences || [];
            pgnFenSequence = data.fen_sequence || [];

            const errCount = (data.errors || []).length;
            const conf     = data.overall_confidence != null
                ? ` · conf <strong>${Math.round(data.overall_confidence * 100)}%</strong>` : '';
            pgnMeta.innerHTML =
                `<strong>${lastGameId}</strong> · Result: <strong>${result}</strong> · ` +
                `${pgnMoves.length} moves` + conf +
                (errCount ? ` · <span style="color:var(--danger);">${errCount} errors</span>` : '');

            // Render board at starting position
            const startFen = pgnFenSequence[0] || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
            renderBoard(startFen);
            pgnMoveLabel.textContent = 'Start';
            renderMoveList(pgnMoves, pgnMoveTags, pgnMoveConfs);
        } else {
            pgnText.value = `Error: ${data.message || 'PGN generation failed'}`;
        }
    } catch (e) {
        pgnText.value = `Error: ${e.message}`;
    }
}

// Navigation buttons
btnGoStart.addEventListener('click', gotoStart);
btnGoEnd.addEventListener('click', () => {
    if (pgnMoves.length > 0) gotoMove(pgnMoves.length - 1);
});
btnGoPrev.addEventListener('click', () => {
    if (currentMoveIdx > 0) gotoMove(currentMoveIdx - 1);
    else gotoStart();
});
btnGoNext.addEventListener('click', () => {
    if (currentMoveIdx < pgnMoves.length - 1) gotoMove(currentMoveIdx + 1);
});

// Keyboard navigation (arrow keys when PGN screen is visible)
document.addEventListener('keydown', e => {
    if (pgnScreen.style.display === 'none') return;
    if (e.key === 'ArrowLeft')  { e.preventDefault(); btnGoPrev.click(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); btnGoNext.click(); }
    if (e.key === 'Home')       { e.preventDefault(); gotoStart(); }
    if (e.key === 'End')        { e.preventDefault(); btnGoEnd.click(); }
});

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
                document.querySelectorAll('.result-btn').forEach(b => b.disabled = true);
                showPgnScreen(result);
            }
        } catch (e) { console.error(e); }
    });
});

btnCopyPgn.addEventListener('click', async () => {
    try {
        await navigator.clipboard.writeText(pgnText.value);
        const orig = btnCopyPgn.innerText;
        btnCopyPgn.innerText = 'Copied ✓';
        setTimeout(() => { btnCopyPgn.innerText = orig; }, 1500);
    } catch {
        pgnText.select();
        document.execCommand('copy');
    }
});

btnDownloadPgn.addEventListener('click', () => {
    const blob = new Blob([pgnText.value], { type: 'application/x-chess-pgn' });
    const url  = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${lastGameId || 'game'}.pgn`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
});

btnNewSameCal.addEventListener('click', async () => {
    btnNewSameCal.disabled = true;
    btnNewSameCal.innerText = 'Starting…';
    try {
        const res = await fetch('/api/new_game_same_calibration', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                white: 'test1', black: 'test2',
                event: '', site: '', game_date: new Date().toISOString().slice(0, 10),
                round: '-', time_control: 'Casual', notes: '', save_raw: false,
            }),
        });
        const data = await res.json();
        if (data.status === 'success') {
            pgnScreen.style.display = 'none';
            populateGameInfo(data.game_info);
            document.querySelectorAll('.result-btn').forEach(b => b.disabled = false);
            updateStatusBadge('STATIC');
            btnNewSameCal.disabled = false;
            btnNewSameCal.innerText = 'New Game (Same Calibration)';
        } else {
            alert('Failed: ' + (data.message || ''));
            btnNewSameCal.disabled = false;
            btnNewSameCal.innerText = 'New Game (Same Calibration)';
        }
    } catch (e) {
        console.error(e);
        btnNewSameCal.disabled = false;
        btnNewSameCal.innerText = 'New Game (Same Calibration)';
    }
});

btnNewFreshCal.addEventListener('click', async () => {
    pgnScreen.style.display = 'none';
    btnReset.click();   // reuse existing full reset flow
});

// ─── Session loop ─────────────────────────────────────────────────────────────
const CAL_STATES = new Set(['CALIBRATING', 'GRID_CORRECTION', 'ORIENTATION']);

function updateStatusBadge(state) {
    statusBadge.innerText = state;
    statusBadge.className = 'status-badge ' + state.toLowerCase();
    const gameMain = document.getElementById('game-panel');
    if (CAL_STATES.has(state)) {
        gameMain.classList.add('cal-fullscreen');
    } else {
        gameMain.classList.remove('cal-fullscreen');
        setCalibrationMenuOpen(false);
    }
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
            const fb64 = getFrameBase64();
            if (!fb64) return;
            const res  = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_b64: fb64 }),
            });
            const data = await res.json();
            if (data.state)    updateStatusBadge(data.state);
            if (data.mask_b64) debugMask.src = data.mask_b64;
            if (data.board_thumb_b64) {
                boardPreviewImg.src = data.board_thumb_b64;
                boardPreviewMove.innerText = `Move ${data.move_number != null ? data.move_number - 1 : '?'}`;
                boardPreviewPanel.style.display = '';
            }
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
    endGamePanel.style.display     = 'none';
    resultDisplay.style.display    = 'none';
    pgnScreen.style.display        = 'none';
    boardPreviewPanel.style.display = 'none';
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
