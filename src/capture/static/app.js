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
const btnClear        = document.getElementById('btn-clear');
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

// ─── State ───────────────────────────────────────────────────────────────────
let calibrationPoints = [];
let isCalibrated      = false;
let sessionLoop       = null;
let selectedA1        = null;

// ─── Setup form ──────────────────────────────────────────────────────────────

// Default date to today
inpDate.value = new Date().toISOString().slice(0, 10);

// Auto-fill Event from player names
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

    btnStartSession.disabled    = true;
    btnStartSession.innerText   = 'Starting…';

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
    infoWhite.innerText  = info.white  || '—';
    infoBlack.innerText  = info.black  || '—';
    infoEvent.innerText  = info.event  || '—';
    infoDate.innerText   = info.date   || '—';
    infoTc.innerText     = info.time_control || '—';
}

// ─── Camera ──────────────────────────────────────────────────────────────────
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' }
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            overlayCanvas.width  = video.videoWidth;
            overlayCanvas.height = video.videoHeight;
        };
    } catch (err) {
        alert('Could not access webcam. Check browser permissions.');
    }
}

function getFrameBase64() {
    const tmp = document.createElement('canvas');
    tmp.width  = video.videoWidth;
    tmp.height = video.videoHeight;
    tmp.getContext('2d').drawImage(video, 0, 0);
    return tmp.toDataURL('image/jpeg', 0.8);
}

// ─── Step 1: Corner calibration ───────────────────────────────────────────────
function drawCornerPoints() {
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    calibrationPoints.forEach((pt, i) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#da3633';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = '16px Inter';
        ctx.fillText(i + 1, pt.x + 10, pt.y - 10);
    });
    if (calibrationPoints.length === 4) {
        ctx.beginPath();
        ctx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
        for (let i = 1; i < 4; i++) ctx.lineTo(calibrationPoints[i].x, calibrationPoints[i].y);
        ctx.closePath();
        ctx.strokeStyle = 'rgba(88, 166, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'rgba(46, 160, 67, 0.2)';
        ctx.fill();
    }
}

overlayCanvas.addEventListener('mousedown', (e) => {
    if (isCalibrated || calibrationPoints.length >= 4) return;
    const rect = overlayCanvas.getBoundingClientRect();
    calibrationPoints.push({
        x: (e.clientX - rect.left)  * (overlayCanvas.width  / rect.width),
        y: (e.clientY - rect.top)   * (overlayCanvas.height / rect.height),
    });
    drawCornerPoints();
    if (calibrationPoints.length === 4) btnCalibrate.disabled = false;
});

btnClear.addEventListener('click', () => {
    calibrationPoints = [];
    btnCalibrate.disabled = true;
    drawCornerPoints();
});

btnCalibrate.addEventListener('click', async () => {
    if (calibrationPoints.length !== 4) return;
    btnCalibrate.disabled = true;
    btnCalibrate.innerText = 'Calibrating…';
    try {
        const res  = await fetch('/api/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: calibrationPoints, image_b64: getFrameBase64() }),
        });
        const data = await res.json();
        if (data.status === 'success') {
            showOrientationStep(data.warped_b64);
        } else {
            alert('Calibration failed!');
            btnCalibrate.innerText = 'Calibrate Board';
            btnCalibrate.disabled  = false;
        }
    } catch (e) {
        console.error(e);
        btnCalibrate.innerText = 'Calibrate Board';
        btnCalibrate.disabled  = false;
    }
});

// ─── Step 2: a1 orientation selection ────────────────────────────────────────
const GRID = 8;

function snapToCorner(col, row) {
    const corners = [{col:0,row:0},{col:7,row:0},{col:0,row:7},{col:7,row:7}];
    return corners.reduce((best, c) => {
        const d  = Math.abs(c.col - col) + Math.abs(c.row - row);
        const bd = Math.abs(best.col - col) + Math.abs(best.row - row);
        return d < bd ? c : best;
    });
}

function drawOrientationGrid(highlighted) {
    const W = orientCanvas.width;
    const H = orientCanvas.height;
    const cw = W / GRID;
    const ch = H / GRID;
    octx.clearRect(0, 0, W, H);

    if (highlighted) {
        octx.fillStyle = 'rgba(46, 160, 67, 0.5)';
        octx.fillRect(highlighted.col * cw, highlighted.row * ch, cw, ch);
        octx.fillStyle = '#fff';
        octx.font = `bold ${Math.round(cw * 0.35)}px Inter`;
        octx.textAlign = 'center';
        octx.textBaseline = 'middle';
        octx.fillText('a1', (highlighted.col + 0.5) * cw, (highlighted.row + 0.5) * ch);
    }

    octx.strokeStyle = 'rgba(88, 166, 255, 0.6)';
    octx.lineWidth = 1;
    for (let i = 1; i < GRID; i++) {
        octx.beginPath(); octx.moveTo(i * cw, 0); octx.lineTo(i * cw, H); octx.stroke();
        octx.beginPath(); octx.moveTo(0, i * ch); octx.lineTo(W, i * ch); octx.stroke();
    }

    const cornerLabels = [
        {col:0,row:0,text:'TL'},{col:7,row:0,text:'TR'},
        {col:0,row:7,text:'BL'},{col:7,row:7,text:'BR'},
    ];
    octx.fillStyle = 'rgba(255,255,255,0.35)';
    octx.font = `${Math.round(cw * 0.22)}px Inter`;
    octx.textAlign = 'center';
    octx.textBaseline = 'middle';
    cornerLabels.forEach(l => {
        if (highlighted && l.col === highlighted.col && l.row === highlighted.row) return;
        octx.fillText(l.text, (l.col + 0.5) * cw, (l.row + 0.5) * ch);
    });
    octx.textAlign = 'left';
    octx.textBaseline = 'alphabetic';
}

function showOrientationStep(warpedB64) {
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
    selectedA1 = null;
    btnConfirmA1.disabled  = true;
    btnConfirmA1.innerText = 'Confirm a1';
    orientHint.innerHTML   = 'Click on the <strong>a1</strong> square (bottom-left from White\'s view)';
    updateStatusBadge('ORIENTATION');
}

orientCanvas.addEventListener('mousedown', (e) => {
    const rect = orientCanvas.getBoundingClientRect();
    const col  = Math.floor((e.clientX - rect.left) / (rect.width  / GRID));
    const row  = Math.floor((e.clientY - rect.top)  / (rect.height / GRID));
    selectedA1 = snapToCorner(col, row);
    btnConfirmA1.disabled = false;
    orientCanvas.width  = warpedImg.naturalWidth;
    orientCanvas.height = warpedImg.naturalHeight;
    drawOrientationGrid(selectedA1);
    const names = {'0,7':'Bottom-Left','7,7':'Bottom-Right','0,0':'Top-Left','7,0':'Top-Right'};
    orientHint.innerHTML = `a1 at <strong>${names[`${selectedA1.col},${selectedA1.row}`]}</strong> — confirm?`;
});

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
    calibrationPoints = [];
    selectedA1 = null;
    btnCalibrate.disabled  = true;
    btnCalibrate.innerText = 'Calibrate Board';
    drawCornerPoints();
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
                resultDisplay.innerText  = `Result recorded: ${result}`;
                resultDisplay.style.display = '';
                document.querySelectorAll('.result-btn').forEach(b => b.disabled = true);
            }
        } catch (e) { console.error(e); }
    });
});

// ─── Session loop ─────────────────────────────────────────────────────────────
function updateStatusBadge(state) {
    statusBadge.innerText   = state;
    statusBadge.className   = 'status-badge ' + state.toLowerCase();
}

function updateUI(data) {
    updateStatusBadge(data.state);
    infoGameId.innerText = data.game_id || '—';
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
            if (data.state) updateStatusBadge(data.state);
            if (data.mask_b64) debugMask.src = data.mask_b64;
            if (Math.random() < 0.2) fetchState();
        } catch (e) { console.error('Frame drop:', e); }
    }, 200);
}

btnReset.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    // Return to setup screen
    if (sessionLoop) clearInterval(sessionLoop);
    isCalibrated = false;
    calibrationPoints = [];
    selectedA1 = null;
    gamePanel.style.display  = 'none';
    setupPanel.style.display = '';
    endGamePanel.style.display  = 'none';
    resultDisplay.style.display = 'none';
    document.querySelectorAll('.result-btn').forEach(b => b.disabled = false);
    btnCalibrate.disabled  = true;
    btnCalibrate.innerText = 'Calibrate Board';
    btnStartSession.disabled  = false;
    btnStartSession.innerText = 'Start Session →';
    delete inpEvent.dataset.userEdited;
    updateStatusBadge('SETUP');
});

// ─── Boot ─────────────────────────────────────────────────────────────────────
fetchState().then(data => {
    if (data.state === 'STATIC' || data.state === 'MOVING') {
        // Reconnected to an active session
        setupPanel.style.display = 'none';
        gamePanel.style.display  = '';
        isCalibrated = true;
        if (data.game_info) populateGameInfo(data.game_info);
        endGamePanel.style.display = '';
        initCamera().then(startSessionLoop);
    }
});
