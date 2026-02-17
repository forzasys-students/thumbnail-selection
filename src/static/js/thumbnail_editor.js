/**
 * Thumbnail Editor 

 */

class ThumbnailEditor {
    constructor(imageFilename, canvasId = 'thumbnail-canvas') {
        this.imageFilename   = imageFilename;
        this.canvas          = document.getElementById(canvasId);
        this.ctx             = this.canvas.getContext('2d');

        // images
        this.originalImage   = null;
        this.maskImage       = null;

        // state
        this.textElements    = [];
        this.backgroundColor = '#000000';
        this.selectedTextId  = null;
        this.isDragging      = false;
        this.dragOffset      = { x: 0, y: 0 };

        // new options
        this.playerLayer     = 'foreground';  // 'foreground' | 'background'
        this.blurBackground  = false;
        this.blurRadius      = 12;

        // internal
        this.maskAvailable      = false;
        this.isSegmenting       = false;
        this._blurCache         = null;   // cached blurred ImageBitmap

        this.canvas.addEventListener('mousedown',  this._onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove',  this._onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup',    this._onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this._onMouseUp.bind(this));
    }

    // =========================================================================
    // Init
    // =========================================================================
    async init() {
        try {
            this.originalImage = await this._loadImg(`/keyframes/${this.imageFilename}`);
            this.canvas.width  = this.originalImage.naturalWidth;
            this.canvas.height = this.originalImage.naturalHeight;
            this._render();
        } catch (e) {
            this.showError('Failed to load image: ' + e.message);
        }
    }

    _loadImg(url) {
        return new Promise((res, rej) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload  = () => res(img);
            img.onerror = () => rej(new Error('Could not load ' + url));
            img.src = url + '?t=' + Date.now(); // cache-bust
        });
    }

    // =========================================================================
    // Segmentation
    // =========================================================================
    async segmentPlayer(prompt = 'soccer player') {
        if (this.isSegmenting) return;
        this.isSegmenting = true;
        this.showStatus('Segmenting… this may take a few seconds');

        try {
            const res  = await fetch('/api/segment-player', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({
                    keyframe_filename: this.imageFilename,
                    prompt: prompt.trim() || 'soccer player'
                })
            });
            const data = await res.json();

            if (!data.success) throw new Error(data.error || 'Segmentation failed');
            if (!data.mask_url) throw new Error('No mask URL returned');

            // Load the mask – mask_url is like "/masks/filename_mask.png"
            this.maskImage     = await this._loadImg(data.mask_url);
            this.maskAvailable = true;
            this._blurCache    = null;  // invalidate

            // Store exact mask filename for download (avoids name re-derivation bugs)
            this._maskFilename = data.mask_url.split('/').pop();

            this._render();
            this.showStatus(`✓ Found ${data.num_persons} instance(s)`);

            // Reveal the player-layer toggle
            const row = document.getElementById('player-layer-row');
            if (row) row.classList.remove('hidden');

        } catch (e) {
            this.showError('Segmentation failed: ' + e.message);
        } finally {
            this.isSegmenting = false;
        }
    }

    // =========================================================================
    // Core render
    //
    // The composition order that matches the inspiration images:
    //
    // playerLayer === 'foreground'  (player pops in FRONT of text):
    //   1. Original photo (or blurred bg)
    //   2. Background-layer text
    //   3. Player cutout
    //   4. Foreground-layer text
    //
    // playerLayer === 'background'  (player sits BEHIND text):
    //   1. Solid colour OR blurred photo
    //   2. Player cutout
    //   3. Background-layer text
    //   4. Foreground-layer text
    // =========================================================================
    _render() {
        const ctx = this.ctx;
        const w   = this.canvas.width;
        const h   = this.canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (!this.originalImage) return;

        const hasMask = this.maskAvailable && !!this.maskImage;

        // ------------------------------------------------------------------
        // BASE LAYER
        // ------------------------------------------------------------------
        if (!hasMask) {
            // No mask yet – just show the photo so the user sees something
            ctx.drawImage(this.originalImage, 0, 0, w, h);

        } else if (this.blurBackground) {
            // Blurred photo background (player will be painted sharp later)
            this._drawBlurredBg(ctx, w, h);

        } else if (this.playerLayer === 'foreground') {
            // Full original photo; player cutout will be composited on top
            ctx.drawImage(this.originalImage, 0, 0, w, h);

        } else {
            // Solid colour (player behind text style)
            ctx.fillStyle = this.backgroundColor;
            ctx.fillRect(0, 0, w, h);
        }

        // ------------------------------------------------------------------
        // PLAYER BEHIND TEXT
        // ------------------------------------------------------------------
        if (hasMask && this.playerLayer === 'background') {
            this._drawPlayerCutout(ctx, w, h);
        }

        // ------------------------------------------------------------------
        // BACKGROUND-LAYER TEXT
        // ------------------------------------------------------------------
        this.textElements
            .filter(t => t.layer === 'background')
            .forEach(t => this._drawText(ctx, t));

        // ------------------------------------------------------------------
        // PLAYER IN FRONT OF TEXT
        // ------------------------------------------------------------------
        if (hasMask && this.playerLayer === 'foreground') {
            this._drawPlayerCutout(ctx, w, h);
        }

        // ------------------------------------------------------------------
        // FOREGROUND-LAYER TEXT  (always on top of everything)
        // ------------------------------------------------------------------
        this.textElements
            .filter(t => t.layer === 'foreground')
            .forEach(t => this._drawText(ctx, t));

        // ------------------------------------------------------------------
        // SELECTION HIGHLIGHT
        // ------------------------------------------------------------------
        if (this.selectedTextId !== null) this._drawSelection(ctx);
    }

    // ── Draw blurred photo background, then paint sharp player back on top ──
    _drawBlurredBg(ctx, w, h) {
        if (!this._blurCache) {
            const off    = document.createElement('canvas');
            off.width    = w;
            off.height   = h;
            const offCtx = off.getContext('2d');

            // Fill with bg color first to prevent black bleed at edges from blur
            offCtx.fillStyle = this.backgroundColor;
            offCtx.fillRect(0, 0, w, h);

            // ctx.filter IS supported in Chrome/Edge/Firefox for 2d canvas
            offCtx.filter = `blur(${this.blurRadius}px)`;
            offCtx.drawImage(this.originalImage, 0, 0, w, h);
            offCtx.filter = 'none';

            this._blurCache = off;
        }
        ctx.drawImage(this._blurCache, 0, 0);
    }

    // ── Paste the segmented player cutout ────────────────────────────────────
    _drawPlayerCutout(ctx, w, h) {
        // The mask PNG from SAM3 is a GREYSCALE image (white=player, black=bg).
        // Canvas destination-in uses the alpha of the mask pixels, NOT brightness.
        // A greyscale PNG loaded as <img> has alpha=255 everywhere → mask is ignored.
        //
        // Fix: manually copy mask brightness into alpha channel of the player image.
        const maskCanvas    = document.createElement('canvas');
        maskCanvas.width    = w;
        maskCanvas.height   = h;
        const maskCtx       = maskCanvas.getContext('2d');
        maskCtx.drawImage(this.maskImage, 0, 0, w, h);
        const maskData      = maskCtx.getImageData(0, 0, w, h).data;

        const tmp    = document.createElement('canvas');
        tmp.width    = w;
        tmp.height   = h;
        const tmpCtx = tmp.getContext('2d');
        tmpCtx.drawImage(this.originalImage, 0, 0, w, h);
        const imgData = tmpCtx.getImageData(0, 0, w, h);

        // Set each pixel's alpha to the mask's red channel (greyscale = R=G=B)
        for (let i = 0; i < imgData.data.length; i += 4) {
            imgData.data[i + 3] = maskData[i]; // R of greyscale = brightness = desired alpha
        }
        tmpCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(tmp, 0, 0);
    }

    // ── Draw text with outline ────────────────────────────────────────────────
    _drawText(ctx, t) {
        const { content, x, y, fontSize, color, strokeColor, strokeWidth, fontFamily } = t;
        ctx.font         = `bold ${fontSize}px ${fontFamily || 'Impact'}, Arial Black, sans-serif`;
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'top';

        if (strokeWidth > 0) {
            ctx.lineJoin    = 'round';
            ctx.strokeStyle = strokeColor || '#000000';
            ctx.lineWidth   = strokeWidth;
            ctx.strokeText(content, x, y);
        }
        ctx.fillStyle = color || '#ffffff';
        ctx.fillText(content, x, y);
    }

    // ── Dashed selection rect around selected text ────────────────────────────
    _drawSelection(ctx) {
        const t = this.textElements.find(el => el.id === this.selectedTextId);
        if (!t) return;
        ctx.font = `bold ${t.fontSize}px ${t.fontFamily || 'Impact'}, Arial Black, sans-serif`;
        const tw = ctx.measureText(t.content).width;
        const th = t.fontSize * 1.2;
        ctx.strokeStyle = '#00cfff';
        ctx.lineWidth   = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(t.x - 6, t.y - 6, tw + 12, th + 12);
        ctx.setLineDash([]);
    }

    // =========================================================================
    // Text management
    // =========================================================================
    addText(content = 'HIGHLIGHTS', x = 80, y = 80, opts = {}) {
        const el = {
            id:          Date.now(),
            content,
            x, y,
            fontSize:    opts.fontSize    ?? 140,
            color:       opts.color       ?? '#ffffff',
            strokeColor: opts.strokeColor ?? '#000000',
            strokeWidth: opts.strokeWidth ?? 8,
            fontFamily:  opts.fontFamily  ?? 'Impact',
            layer:       opts.layer       ?? 'background'
        };
        this.textElements.push(el);
        this.selectedTextId = el.id;
        this._render();
        this._emitTextList();
        return el.id;
    }

    updateText(id, changes) {
        const el = this.textElements.find(t => t.id === id);
        if (!el) return;
        Object.assign(el, changes);
        this._render();
        this._emitTextList();
    }

    deleteText(id) {
        this.textElements = this.textElements.filter(t => t.id !== id);
        if (this.selectedTextId === id) this.selectedTextId = null;
        this._render();
        this._emitTextList();
    }

    // =========================================================================
    // Mouse – scale-corrected so drag works when canvas is CSS-scaled
    // =========================================================================
    _canvasXY(e) {
        const r  = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (this.canvas.width  / r.width),
            y: (e.clientY - r.top)  * (this.canvas.height / r.height)
        };
    }

    _textHit(px, py, t) {
        this.ctx.font = `bold ${t.fontSize}px ${t.fontFamily || 'Impact'}, Arial Black, sans-serif`;
        const w = this.ctx.measureText(t.content).width;
        const h = t.fontSize * 1.2;
        return px >= t.x && px <= t.x + w && py >= t.y && py <= t.y + h;
    }

    _onMouseDown(e) {
        const { x, y } = this._canvasXY(e);
        for (const t of [...this.textElements].reverse()) {
            if (this._textHit(x, y, t)) {
                this.selectedTextId = t.id;
                this.isDragging     = true;
                this.dragOffset     = { x: x - t.x, y: y - t.y };
                this._render();
                this._emitTextList();
                return;
            }
        }
        this.selectedTextId = null;
        this._render();
        this._emitTextList();
    }

    _onMouseMove(e) {
        const { x, y } = this._canvasXY(e);
        if (this.isDragging && this.selectedTextId) {
            const t = this.textElements.find(el => el.id === this.selectedTextId);
            if (t) { t.x = x - this.dragOffset.x; t.y = y - this.dragOffset.y; this._render(); }
        } else {
            const over = this.textElements.some(t => this._textHit(x, y, t));
            this.canvas.style.cursor = over ? 'move' : 'default';
        }
    }

    _onMouseUp() { this.isDragging = false; }

    // =========================================================================
    // Download
    // =========================================================================
    async downloadThumbnail() {
        this.showStatus('Rendering final thumbnail…');
        try {
            // Use the exact mask filename returned by the segmentation API,
            // not a re-derived name (avoids mismatch if stem differs).
            const maskFilename = this._maskFilename || null;

            const res = await fetch('/api/create-thumbnail', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    keyframe_filename: this.imageFilename,
                    mask_filename:     maskFilename,
                    text_elements:     this.textElements,
                    background_color:  this.backgroundColor,
                    player_layer:      this.playerLayer,
                    blur_background:   this.blurBackground,
                    blur_radius:       this.blurRadius
                })
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error);

            const a = document.createElement('a');
            a.href     = data.thumbnail_url + '?t=' + Date.now();
            a.download = `thumbnail_${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            this.showStatus('Downloaded!');
        } catch (e) {
            this.showError('Download failed: ' + e.message);
        }
    }

    // =========================================================================
    // Events & status
    // =========================================================================
    _emitTextList() {
        document.dispatchEvent(new CustomEvent('textListChanged', {
            detail: { textElements: this.textElements, selectedId: this.selectedTextId }
        }));
    }

    showStatus(msg) {
        document.dispatchEvent(new CustomEvent('editorStatus', { detail: { msg, type: 'info' } }));
    }

    showError(msg) {
        document.dispatchEvent(new CustomEvent('editorStatus', { detail: { msg, type: 'error' } }));
    }
}


// =============================================================================
// UI  (modal, controls, event wiring)
// =============================================================================

function openThumbnailEditor(filename) {
    let modal = document.getElementById('te-modal');
    if (!modal) { modal = _buildModal(); document.body.appendChild(modal); }

    // Reset every time it opens
    modal.style.display = 'flex';
    const editor = new ThumbnailEditor(filename, 'te-canvas');
    window._editor = editor;
    editor.init();
    _wireControls(editor);
}

function closeThumbnailEditor() {
    const m = document.getElementById('te-modal');
    if (m) m.style.display = 'none';
    window._editor = null;
}

// Expose helpers called from inline onclick
function teSegment()        { const p = document.getElementById('te-prompt'); window._editor.segmentPlayer(p ? p.value : ''); }
function teAddText()        { const v = prompt('Enter text:', 'HIGHLIGHTS'); if (v) window._editor.addText(v); }
function teDelText(id)      { window._editor.deleteText(id); }
function teProp(id,k,v)     { window._editor.updateText(id, {[k]: v}); }
function teBgColor(v)       { window._editor.backgroundColor = v; window._editor._render(); }
function teDownload()       { window._editor.downloadThumbnail(); }

function tePlayerLayer(layer) {
    window._editor.playerLayer = layer;
    window._editor._render();
    document.getElementById('te-pl-bg').classList.toggle('active', layer === 'background');
    document.getElementById('te-pl-fg').classList.toggle('active', layer === 'foreground');
}

function teBlur(checked) {
    window._editor.blurBackground  = checked;
    window._editor._blurCache      = null;
    window._editor._render();
    const row = document.getElementById('te-blur-row');
    if (row) row.classList.toggle('hidden', !checked);
}

function teBlurRadius(v) {
    window._editor.blurRadius  = parseInt(v);
    window._editor._blurCache  = null;
    window._editor._render();
    const d = document.getElementById('te-blur-val');
    if (d) d.textContent = v + 'px';
}

// ── Build modal DOM ───────────────────────────────────────────────────────────
function _buildModal() {
    const m = document.createElement('div');
    m.id        = 'te-modal';
    m.className = 'thumbnail-modal';
    m.innerHTML = `
    <div class="modal-backdrop" onclick="closeThumbnailEditor()"></div>
    <div class="modal-content-wrapper">

      <div class="modal-header">
        <h2>✦ Thumbnail Editor</h2>
        <button class="btn-close" onclick="closeThumbnailEditor()">&times;</button>
      </div>

      <div class="editor-container">

        <!-- Canvas -->
        <div class="canvas-panel">
          <canvas id="te-canvas"></canvas>
          <div id="te-status" class="status-message"></div>
        </div>

        <!-- Controls -->
        <div class="controls-panel">

          <!-- Segmentation -->
          <section class="control-section">
            <h3>Segment</h3>
            <label class="field-label">Prompt</label>
            <input id="te-prompt" class="prompt-input" type="text"
                   value="soccer player"
                   placeholder="e.g. goalkeeper, celebrating player…" />
            <button class="btn-primary" onclick="teSegment()">⬡ Run Segmentation</button>

            <!-- Player layer – hidden until mask ready -->
            <div id="te-layer-row" class="player-layer-row hidden">
              <label class="field-label">Player layer</label>
              <div class="layer-toggle-group">
                <button id="te-pl-bg" class="layer-toggle-btn"       onclick="tePlayerLayer('background')">Behind text</button>
                <button id="te-pl-fg" class="layer-toggle-btn active" onclick="tePlayerLayer('foreground')">In front of text</button>
              </div>
            </div>

            <!-- Blur -->
            <label class="checkbox-row">
              <input type="checkbox" id="te-blur-chk" onchange="teBlur(this.checked)" />
              <span class="checkbox-label">Blur background</span>
            </label>
            <div id="te-blur-row" class="blur-strength-row hidden">
              <label class="field-label">Strength <span id="te-blur-val">12px</span></label>
              <input type="range" class="blur-slider" min="2" max="30" value="12"
                     oninput="teBlurRadius(this.value)" />
            </div>
          </section>

          <!-- Text -->
          <section class="control-section">
            <h3>Text</h3>
            <button class="btn-secondary" onclick="teAddText()">+ Add Text</button>
            <div id="te-textlist"></div>
          </section>

          <!-- Background -->
          <section class="control-section">
            <h3>Background colour</h3>
            <label class="color-row">
              Color <input type="color" id="te-bg-color" value="#000000" onchange="teBgColor(this.value)" />
            </label>
          </section>

          <!-- Download -->
          <section class="control-section">
            <button class="btn-success" onclick="teDownload()">↓ Download Thumbnail</button>
          </section>

        </div><!-- /controls -->
      </div><!-- /editor-container -->
    </div><!-- /modal-content-wrapper -->`;
    return m;
}

// ── Wire control events ───────────────────────────────────────────────────────
function _wireControls(editor) {
    // Remove and re-add listeners to avoid duplicates on re-open
    document.removeEventListener('textListChanged', _onTextListChanged);
    document.removeEventListener('editorStatus',    _onEditorStatus);
    document.addEventListener('textListChanged', _onTextListChanged);
    document.addEventListener('editorStatus',    _onEditorStatus);

    // Reset UI state
    const layerRow = document.getElementById('te-layer-row');
    if (layerRow) layerRow.classList.add('hidden');
    const blurChk  = document.getElementById('te-blur-chk');
    if (blurChk)  blurChk.checked = false;
    const blurRow  = document.getElementById('te-blur-row');
    if (blurRow)  blurRow.classList.add('hidden');
    const textList = document.getElementById('te-textlist');
    if (textList) textList.innerHTML = '<p class="empty-text-list">No text yet</p>';
}

function _onEditorStatus(e) {
    const el = document.getElementById('te-status');
    if (!el) return;
    el.textContent = e.detail.msg;
    el.className   = 'status-message ' + e.detail.type;
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.textContent = ''; el.className = 'status-message'; }, 4000);
}

function _onTextListChanged(e) {
    const list = document.getElementById('te-textlist');
    if (!list) return;
    const { textElements, selectedId } = e.detail;

    if (!textElements.length) {
        list.innerHTML = '<p class="empty-text-list">No text yet</p>';
        return;
    }

    list.innerHTML = '';
    textElements.forEach(t => {
        const div = document.createElement('div');
        div.className = 'text-item' + (t.id === selectedId ? ' selected' : '');
        div.innerHTML = `
          <div class="text-item-header">
            <strong>${t.content}</strong>
            <button class="btn-delete" onclick="teDelText(${t.id})">&times;</button>
          </div>
          <div class="text-item-controls">
            <label>Text
              <input type="text" value="${t.content.replace(/"/g,'&quot;')}"
                     onchange="teProp(${t.id},'content',this.value)" />
            </label>
            <label>Size
              <input type="number" value="${t.fontSize}" min="10" max="500"
                     onchange="teProp(${t.id},'fontSize',+this.value)" />
            </label>
            <label>Colour
              <input type="color" value="${t.color}"
                     onchange="teProp(${t.id},'color',this.value)" />
            </label>
            <label>Outline
              <input type="color" value="${t.strokeColor}"
                     onchange="teProp(${t.id},'strokeColor',this.value)" />
            </label>
            <label>Layer
              <select onchange="teProp(${t.id},'layer',this.value)">
                <option value="background" ${t.layer==='background'?'selected':''}>Behind player</option>
                <option value="foreground" ${t.layer==='foreground'?'selected':''}>In front of player</option>
              </select>
            </label>
          </div>`;
        list.appendChild(div);
    });
}