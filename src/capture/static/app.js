const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');
const btnCalibrate = document.getElementById('btn-calibrate');
const btnClear = document.getElementById('btn-clear');
const btnReset = document.getElementById('btn-reset');
const controlsOverlay = document.getElementById('calibration-controls');

const statusBadge = document.getElementById('app-status');
const infoGameId = document.getElementById('info-game-id');
const infoMoves = document.getElementById('info-moves');
const infoRotation = document.getElementById('info-rotation');
const debugMask = document.getElementById('debug-mask');

let calibrationPoints = [];
let isCalibrated = false;
let sessionLoop = null;

// Initialize Webcam
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "environment" // Prefer external/back camera if mobile
            } 
        });
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access webcam. Please ensure permissions are granted.");
    }
}

// Draw overlaid points
function drawPoints() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    calibrationPoints.forEach((pt, index) => {
        // Draw Dot
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = '#da3633';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw Number
        ctx.fillStyle = '#fff';
        ctx.font = '16px Inter';
        ctx.fillText(index + 1, pt.x + 10, pt.y - 10);
    });
    
    // Connect points with lines if we have 4
    if (calibrationPoints.length === 4) {
        ctx.beginPath();
        ctx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
        ctx.lineTo(calibrationPoints[1].x, calibrationPoints[1].y);
        ctx.lineTo(calibrationPoints[2].x, calibrationPoints[2].y);
        ctx.lineTo(calibrationPoints[3].x, calibrationPoints[3].y);
        ctx.closePath();
        ctx.strokeStyle = 'rgba(88, 166, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();
        // Fill semi-transparent green inside
        ctx.fillStyle = 'rgba(46, 160, 67, 0.2)';
        ctx.fill();
    }
}

// Capture Video Frame to Base64
function getFrameBase64() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    tempCanvas.getContext('2d').drawImage(video, 0, 0);
    return tempCanvas.toDataURL('image/jpeg', 0.8);
}

// Canvas Click handler
canvas.addEventListener('mousedown', (e) => {
    if (isCalibrated) return;
    
    if (calibrationPoints.length < 4) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        calibrationPoints.push({x, y});
        drawPoints();
        
        if (calibrationPoints.length === 4) {
            btnCalibrate.disabled = false;
        }
    }
});

btnClear.addEventListener('click', () => {
    calibrationPoints = [];
    btnCalibrate.disabled = true;
    drawPoints();
});

btnReset.addEventListener('click', async () => {
    await fetch('/api/reset', { method: 'POST' });
    isCalibrated = false;
    calibrationPoints = [];
    btnCalibrate.disabled = true;
    controlsOverlay.style.opacity = '1';
    controlsOverlay.style.pointerEvents = 'all';
    drawPoints();
    if(sessionLoop) clearInterval(sessionLoop);
    updateUI({ state: "CALIBRATING", game_id: "--", move_number: 0, rotation_angle: 0 });
});

btnCalibrate.addEventListener('click', async () => {
    if (calibrationPoints.length !== 4) return;
    
    btnCalibrate.disabled = true;
    btnCalibrate.innerText = "Calibrating...";
    
    try {
        const image_b64 = getFrameBase64();
        const res = await fetch('/api/calibrate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ points: calibrationPoints, image_b64 })
        });
        
        const data = await res.json();
        
        if (data.status === 'success') {
            isCalibrated = true;
            controlsOverlay.style.opacity = '0';
            controlsOverlay.style.pointerEvents = 'none';
            ctx.clearRect(0, 0, canvas.width, canvas.height); // hide dots
            startSessionLoop();
        } else {
            alert("Calibration failed!");
            btnCalibrate.innerText = "Calibrate Board";
            btnCalibrate.disabled = false;
        }
    } catch (e) {
        console.error(e);
        btnCalibrate.innerText = "Calibrate Board";
        btnCalibrate.disabled = false;
    }
});

function updateUI(stateData) {
    statusBadge.innerText = stateData.state;
    statusBadge.className = "status-badge " + stateData.state.toLowerCase();
    
    infoGameId.innerText = stateData.game_id;
    infoMoves.innerText = stateData.move_number > 0 ? stateData.move_number - 1 : 0;
    
    if (stateData.rotation_angle !== undefined) {
        infoRotation.innerHTML = `${stateData.rotation_angle}&deg;`;
    }
}

async function fetchState() {
    const res = await fetch('/api/state');
    const data = await res.json();
    updateUI(data);
    return data;
}

// Master loop pushing frames
function startSessionLoop() {
    if(sessionLoop) clearInterval(sessionLoop);
    
    fetchState(); // get Initial ID
    
    // ~5 FPS is enough for motion detection without lagging the browser
    sessionLoop = setInterval(async () => {
        if (!isCalibrated) return;
        
        try {
            const image_b64 = getFrameBase64();
            const res = await fetch('/api/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_b64 })
            });
            
            const data = await res.json();
            
            if (data.state) {
                // UI updates via response
                statusBadge.innerText = data.state;
                statusBadge.className = "status-badge " + data.state.toLowerCase();
                
                if (data.mask_b64) {
                    debugMask.src = data.mask_b64;
                }
            }
            
            // Poll full state occasionally for move checks
            if (Math.random() < 0.2) fetchState();
            
        } catch(e) {
            console.error("Frame drop: ", e);
        }
    }, 200);
}

// Boot
initCamera();
fetchState().then(data => {
    if (data.calibrated) {
        isCalibrated = true;
        controlsOverlay.style.opacity = '0';
        controlsOverlay.style.pointerEvents = 'none';
        startSessionLoop();
    }
});
