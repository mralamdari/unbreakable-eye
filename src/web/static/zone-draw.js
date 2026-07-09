/**
 * Zone drawing overlay for the video wall.
 * Allows drawing polygons on camera tiles to define zones.
 */

(function () {
  let drawingMode = false;
  let currentPoints = [];
  let activeTile = null;
  let canvas = null;
  let ctx = null;

  function initZoneDraw() {
    // Add "Draw Zone" button to each tile
    document.querySelectorAll('.tile').forEach(tile => {
      const header = tile.querySelector('.tile-header');
      if (!header) return;

      const btn = document.createElement('button');
      btn.className = 'zone-draw-btn';
      btn.textContent = '+ Zone';
      btn.onclick = (e) => {
        e.stopPropagation();
        startDrawing(tile);
      };
      header.appendChild(btn);
    });
  }

  function startDrawing(tile) {
    if (drawingMode) stopDrawing();

    drawingMode = true;
    activeTile = tile;
    currentPoints = [];

    const img = tile.querySelector('img');
    if (!img) return;

    // Create canvas overlay
    canvas = document.createElement('canvas');
    canvas.className = 'zone-canvas';
    canvas.width = img.clientWidth;
    canvas.height = img.clientHeight;
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.cursor = 'crosshair';
    canvas.style.zIndex = '20';
    tile.appendChild(canvas);

    ctx = canvas.getContext('2d');

    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('dblclick', onCanvasDblClick);
    document.addEventListener('keydown', onEscape);

    // Show prompt
    showToast('Click to add points. Double-click to finish. Esc to cancel.');
  }

  function onCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    currentPoints.push([x, y]);
    drawPreview();
  }

  function onCanvasDblClick(e) {
    e.preventDefault();
    if (currentPoints.length >= 3) {
      finishDrawing();
    }
  }

  function onEscape(e) {
    if (e.key === 'Escape' && drawingMode) {
      stopDrawing();
    }
  }

  function drawPreview() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (currentPoints.length < 2) {
      // Draw dots
      currentPoints.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * canvas.width, y * canvas.height, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#4f8cff';
        ctx.fill();
      });
      return;
    }

    // Draw polygon outline
    ctx.beginPath();
    ctx.moveTo(currentPoints[0][0] * canvas.width, currentPoints[0][1] * canvas.height);
    for (let i = 1; i < currentPoints.length; i++) {
      ctx.lineTo(currentPoints[i][0] * canvas.width, currentPoints[i][1] * canvas.height);
    }
    ctx.strokeStyle = '#4f8cff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw dots
    currentPoints.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x * canvas.width, y * canvas.height, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#4f8cff';
      ctx.fill();
    });
  }

  async function finishDrawing() {
    const zoneName = prompt('Zone name:', 'Zone ' + (currentPoints.length));
    if (!zoneName) {
      stopDrawing();
      return;
    }

    const camId = activeTile.dataset.camId;
    if (!camId) {
      showToast('Could not determine camera ID');
      stopDrawing();
      return;
    }

    try {
      const resp = await fetch(`/api/zones/${camId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: zoneName,
          polygon: currentPoints,
          zone_type: 'area',
          color: '#4f8cff'
        })
      });

      if (resp.ok) {
        showToast(`Zone "${zoneName}" created`);
      } else {
        const err = await resp.json();
        showToast(`Error: ${err.detail || 'Failed to create zone'}`);
      }
    } catch (e) {
      showToast('Network error');
    }

    stopDrawing();
  }

  function stopDrawing() {
    drawingMode = false;
    currentPoints = [];
    activeTile = null;

    if (canvas) {
      canvas.removeEventListener('click', onCanvasClick);
      canvas.removeEventListener('dblclick', onCanvasDblClick);
      canvas.remove();
      canvas = null;
      ctx = null;
    }

    document.removeEventListener('keydown', onEscape);
  }

  function showToast(msg) {
    let toast = document.getElementById('zone-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'zone-toast';
      toast.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1a2030;color:#e8eaed;padding:10px 20px;border-radius:8px;border:1px solid #2a3a5a;z-index:1000;font-size:14px;';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, 3000);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initZoneDraw);
  } else {
    initZoneDraw();
  }
})();
