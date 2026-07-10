/**
 * Zone overlay and drawing for the dedicated camera page (static frame).
 */

(function () {
  let zones = [];
  let drawingMode = false;
  let currentPoints = [];

  const frame = document.getElementById('camera-frame');
  const overlayCanvas = document.getElementById('zone-overlay');
  const overlayCtx = overlayCanvas.getContext('2d');
  const drawCanvas = document.getElementById('zone-drawing');
  const drawCtx = drawCanvas.getContext('2d');
  const zoneList = document.getElementById('zone-list');
  const toggleBtn = document.getElementById('zone-toggle');

  // ── Canvas sizing ───────────────────────────────────────────────────────

  function setCanvasRect(x, y, w, h) {
    [overlayCanvas, drawCanvas].forEach(function (c) {
      c.width = w;
      c.height = h;
      c.style.width = w + 'px';
      c.style.height = h + 'px';
      c.style.left = x + 'px';
      c.style.top = y + 'px';
    });
    drawOverlay();
  }

  function resizeCanvases() {
    const rect = frame.getBoundingClientRect();
    const containerW = rect.width;
    const containerH = rect.height;
    if (!containerW || !containerH) return;

    const imgW = frame.naturalWidth || frame.width;
    const imgH = frame.naturalHeight || frame.height;

    if (!imgW || !imgH) {
      setCanvasRect(0, 0, containerW, containerH);
      return;
    }

    // Calculate actual visible image bounds (accounting for object-fit: contain)
    const imgRatio = imgW / imgH;
    const containerRatio = containerW / containerH;

    let visibleW, visibleH, offsetX, offsetY;
    if (imgRatio > containerRatio) {
      visibleW = containerW;
      visibleH = containerW / imgRatio;
      offsetX = 0;
      offsetY = (containerH - visibleH) / 2;
    } else {
      visibleH = containerH;
      visibleW = containerH * imgRatio;
      offsetX = (containerW - visibleW) / 2;
      offsetY = 0;
    }

    setCanvasRect(Math.round(offsetX), Math.round(offsetY), Math.round(visibleW), Math.round(visibleH));
  }

  // ── Zone overlay ────────────────────────────────────────────────────────

  function drawOverlay() {
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    overlayCtx.clearRect(0, 0, w, h);

    zones.forEach((zone, idx) => {
      const poly = zone.polygon;
      if (!poly || poly.length < 3) return;
      const color = zone.color || '#4f8cff';

      // Fill polygon
      overlayCtx.beginPath();
      overlayCtx.moveTo(poly[0][0] * w, poly[0][1] * h);
      for (let i = 1; i < poly.length; i++) {
        overlayCtx.lineTo(poly[i][0] * w, poly[i][1] * h);
      }
      overlayCtx.closePath();
      overlayCtx.fillStyle = hexToRgba(color, 0.15);
      overlayCtx.fill();

      // Border
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 2;
      overlayCtx.stroke();

      // Zone number badge at centroid
      const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length * w;
      const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length * h;
      const label = String(idx + 1);

      overlayCtx.beginPath();
      overlayCtx.arc(cx, cy, 12, 0, Math.PI * 2);
      overlayCtx.fillStyle = color;
      overlayCtx.fill();

      overlayCtx.fillStyle = '#fff';
      overlayCtx.font = 'bold 12px sans-serif';
      overlayCtx.textAlign = 'center';
      overlayCtx.textBaseline = 'middle';
      overlayCtx.fillText(label, cx, cy);
    });
  }

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  // ── Zone list sidebar ───────────────────────────────────────────────────

  function renderZoneList() {
    zoneList.innerHTML = '';
    if (zones.length === 0) {
      zoneList.innerHTML = '<p class="zone-empty">No zones yet. Click "+ Zone" to draw one.</p>';
      return;
    }
    zones.forEach((zone, idx) => {
      const div = document.createElement('div');
      div.className = 'zone-item';
      div.innerHTML =
        '<span class="zone-badge" style="background:' + (zone.color || '#4f8cff') + '">' + (idx + 1) + '</span>' +
        '<span class="zone-name">' + escapeHtml(zone.name) + '</span>' +
        '<button class="zone-delete-btn" data-id="' + zone.id + '">&times;</button>';
      zoneList.appendChild(div);
    });

    zoneList.querySelectorAll('.zone-delete-btn').forEach(btn => {
      btn.onclick = function () { deleteZone(parseInt(btn.dataset.id)); };
    });
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── Zone drawing ────────────────────────────────────────────────────────

  window.toggleDrawing = function () {
    if (drawingMode) {
      stopDrawing();
    } else {
      startDrawing();
    }
  };

  function startDrawing() {
    drawingMode = true;
    currentPoints = [];
    toggleBtn.textContent = 'Cancel';
    toggleBtn.classList.add('active');
    drawCanvas.style.display = 'block';
    drawCanvas.addEventListener('click', onDrawClick);
    drawCanvas.addEventListener('dblclick', onDrawDblClick);
    document.addEventListener('keydown', onEscape);
    showToast('Click to add points. Double-click to finish. Esc to cancel.');
  }

  function stopDrawing() {
    drawingMode = false;
    currentPoints = [];
    toggleBtn.textContent = '+ Zone';
    toggleBtn.classList.remove('active');
    drawCanvas.style.display = 'none';
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    drawCanvas.removeEventListener('click', onDrawClick);
    drawCanvas.removeEventListener('dblclick', onDrawDblClick);
    document.removeEventListener('keydown', onEscape);
  }

  function onDrawClick(e) {
    e.stopPropagation();
    var rect = drawCanvas.getBoundingClientRect();
    var x = (e.clientX - rect.left) / rect.width;
    var y = (e.clientY - rect.top) / rect.height;
    currentPoints.push([x, y]);
    drawPreview();
  }

  function onDrawDblClick(e) {
    e.stopPropagation();
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
    var w = drawCanvas.width;
    var h = drawCanvas.height;
    drawCtx.clearRect(0, 0, w, h);

    // Draw polygon lines
    if (currentPoints.length >= 2) {
      drawCtx.beginPath();
      drawCtx.moveTo(currentPoints[0][0] * w, currentPoints[0][1] * h);
      for (var i = 1; i < currentPoints.length; i++) {
        drawCtx.lineTo(currentPoints[i][0] * w, currentPoints[i][1] * h);
      }
      drawCtx.strokeStyle = '#4f8cff';
      drawCtx.lineWidth = 2;
      drawCtx.stroke();
    }

    // Draw points
    currentPoints.forEach(function (p) {
      drawCtx.beginPath();
      drawCtx.arc(p[0] * w, p[1] * h, 5, 0, Math.PI * 2);
      drawCtx.fillStyle = '#4f8cff';
      drawCtx.fill();
    });
  }

  async function finishDrawing() {
    var zoneName = prompt('Zone name:', 'Zone ' + (zones.length + 1));
    if (!zoneName) {
      stopDrawing();
      return;
    }

    try {
      var resp = await fetch('/api/zones/' + CAM_ID, {
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
        showToast('Zone "' + zoneName + '" created');
        await loadZones();
      } else {
        var err = await resp.json();
        showToast('Error: ' + (err.detail || 'Failed to create zone'));
      }
    } catch (e) {
      showToast('Network error');
    }

    stopDrawing();
  }

  // ── Zone deletion ───────────────────────────────────────────────────────

  async function deleteZone(zoneId) {
    if (!confirm('Delete this zone?')) return;

    try {
      var resp = await fetch('/api/zones/' + zoneId, { method: 'DELETE' });
      if (resp.ok) {
        showToast('Zone deleted');
        await loadZones();
      } else {
        showToast('Failed to delete zone');
      }
    } catch (e) {
      showToast('Network error');
    }
  }

  // ── Data loading ────────────────────────────────────────────────────────

  async function loadZones() {
    try {
      var resp = await fetch('/api/zones/' + CAM_ID);
      if (resp.ok) {
        var data = await resp.json();
        zones = data.zones || [];
      }
    } catch (e) {
      // keep existing zones
    }
    drawOverlay();
    renderZoneList();
  }

  // ── Toast ───────────────────────────────────────────────────────────────

  function showToast(msg) {
    var toast = document.getElementById('zone-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.id = 'zone-toast';
      toast.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1a2030;color:#e8eaed;padding:10px 20px;border-radius:8px;border:1px solid #2a3a5a;z-index:1000;font-size:14px;';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(function () { toast.style.display = 'none'; }, 3000);
  }

  // ── Init ────────────────────────────────────────────────────────────────

  frame.addEventListener('load', resizeCanvases);
  window.addEventListener('resize', resizeCanvases);

  zones = INITIAL_ZONES;

  // Wait for image to be ready
  if (frame.complete) {
    resizeCanvases();
  }

  drawOverlay();
  renderZoneList();
})();
