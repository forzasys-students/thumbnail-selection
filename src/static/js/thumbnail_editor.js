class ThumbnailEditor {
    constructor(imageFilename, canvasId = 'te-canvas') {
        this.imageFilename = imageFilename;
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        // Images
        this.originalImage = null;
        this.maskImage = null;
        this.loadedLogos = {}; // cache for team logos

        // Elements (unified array for all graphic elements)
        this.elements = []; // text, shapes, logos - all elements
        this.selectedId = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };

        // Options
        this.backgroundColor = '#000000';
        this.playerLayer = 'foreground';
        this.blurBackground = false;
        this.blurRadius = 12;

        // Metadata
        this.metadata = null; // will hold game metadata

        // Internal
        this.maskAvailable = false;
        this.isSegmenting = false;
        this._blurCache = null;
        this._maskFilename = null;
        this._idCounter = 1; // unique ID for every element;

        // Event listeners
        this.canvas.addEventListener('mousedown', this._onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this._onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this._onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this._onMouseUp.bind(this));
    }

    // =========================================================================
    // Init
    // =========================================================================
    async init(metadata = null) {
        this.metadata = null; 
        try {
            this.originalImage = await this._loadImg(`/keyframes/${this.imageFilename}`);
            this.canvas.width = this.originalImage.naturalWidth;
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
            img.onload = () => res(img);
            img.onerror = () => rej(new Error('Could not load ' + url));
            img.src = url + '?t=' + Date.now();
        });
    }

    // =========================================================================
    // Segmentation
    // =========================================================================
    async segmentPlayer(prompt = 'soccer player') {
        if (this.isSegmenting) return;
        this.isSegmenting = true;
        this.showStatus('Segmenting…');

        try {
            const res = await fetch('/api/segment-player', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    keyframe_filename: this.imageFilename,
                    prompt: prompt.trim() || 'soccer player'
                })
            });
            const data = await res.json();

            if (!data.success) throw new Error(data.error || 'Segmentation failed');
            if (!data.mask_url) throw new Error('No mask URL returned');

            this.maskImage = await this._loadImg(data.mask_url);
            this.maskAvailable = true;
            this._blurCache = null;
            this._maskFilename = data.mask_url.split('/').pop();

            this._render();
            this.showStatus(`✓ Found ${data.num_persons} instance(s)`);

            const row = document.getElementById('te-layer-row');
            if (row) row.classList.remove('hidden');

        } catch (e) {
            this.showError('Segmentation failed: ' + e.message);
        } finally {
            this.isSegmenting = false;
        }
    }

    async _ensureMetadataLoaded() {
        if (this.metadata) return;

        this.showStatus('Loading metadata…');

        try {
            const res = await fetch(`/api/video-metadata/${encodeURIComponent(this.imageFilename)}`);
            const data = await res.json();

            if (!data.success || !data.metadata) {
                throw new Error('No metadata found for this frame');
            }

            this.metadata = data.metadata;

            this.showStatus('✓ Metadata loaded');

            this._updateMetadataUI();

        } catch (e) {
            this.metadata = null; // important: reset
            this.showError('Metadata load failed: ' + e.message);
        }
    }
    
    // =========================================================================
    // Render Pipeline
    // =========================================================================
    _render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        ctx.clearRect(0, 0, w, h);

        if (!this.originalImage) return;

        const hasMask = this.maskAvailable && !!this.maskImage;

        // Base layer
        if (!hasMask) {
            ctx.drawImage(this.originalImage, 0, 0, w, h);
        } else if (this.blurBackground) {
            this._drawBlurredBg(ctx, w, h);
        } else if (this.playerLayer === 'foreground') {
            ctx.drawImage(this.originalImage, 0, 0, w, h);
        } else {
            ctx.fillStyle = this.backgroundColor;
            ctx.fillRect(0, 0, w, h);
        }

        // Player behind text
        if (hasMask && this.playerLayer === 'background') {
            this._drawPlayerCutout(ctx, w, h);
        }

        // Background elements
        this.elements
            .filter(e => e.layer === 'background')
            .forEach(e => this._drawElement(ctx, e));

        // Player in front of background elements
        if (hasMask && this.playerLayer === 'foreground') {
            this._drawPlayerCutout(ctx, w, h);
        }

        // Foreground elements
        this.elements
            .filter(e => e.layer === 'foreground')
            .forEach(e => this._drawElement(ctx, e));

        // Selection highlight
        if (this.selectedId) this._drawSelection(ctx);
    }

    // ── Draw any element (text, shape, logo) ─────────────────────────────────
    _drawElement(ctx, el) {
        switch (el.type) {
            case 'text': this._drawText(ctx, el); break;
            case 'rect': this._drawRect(ctx, el); break;
            case 'logo': this._drawLogo(ctx, el); break;
        }
    }

    // ── Text ──────────────────────────────────────────────────────────────────
    _drawText(ctx, el) {
        const { content, x, y, fontSize, color, strokeColor, strokeWidth, fontFamily, fontWeight } = el;
        ctx.font = `${fontWeight || 'bold'} ${fontSize}px ${fontFamily || 'Impact'}, Arial Black, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        if (strokeWidth > 0) {
            ctx.lineJoin = 'round';
            ctx.strokeStyle = strokeColor || '#000000';
            ctx.lineWidth = strokeWidth;
            ctx.strokeText(content, x, y);
        }
        ctx.fillStyle = color || '#ffffff';
        ctx.fillText(content, x, y);
    }

    // ── Rectangle ─────────────────────────────────────────────────────────────
    _drawRect(ctx, el) {
        const { x, y, width, height, fillColor, strokeColor, strokeWidth, cornerRadius } = el;

        if (cornerRadius > 0) {
            ctx.beginPath();
            ctx.roundRect(x, y, width, height, cornerRadius);
            ctx.closePath();
        } else {
            ctx.beginPath();
            ctx.rect(x, y, width, height);
            ctx.closePath();
        }

        ctx.save(); 
        if (fillColor) {
            ctx.globalAlpha = el.opacity ?? 1.0;   
            ctx.fillStyle = fillColor;
            ctx.fill();
        }
        ctx.restore(); 

        if (strokeWidth > 0 && strokeColor) {
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = strokeWidth;
            ctx.stroke();
        }
    }

    // ── Logo ──────────────────────────────────────────────────────────────────
    _drawLogo(ctx, el) {
        const { x, y, width, height, imageUrl } = el;
        const img = this.loadedLogos[imageUrl];
        if (img && img.complete) {
            ctx.drawImage(img, x, y, width, height);
        }
    }

    // ── Blurred background ────────────────────────────────────────────────────
    _drawBlurredBg(ctx, w, h) {
        if (!this._blurCache) {
            const off = document.createElement('canvas');
            off.width = w;
            off.height = h;
            const offCtx = off.getContext('2d');
            offCtx.fillStyle = this.backgroundColor;
            offCtx.fillRect(0, 0, w, h);
            offCtx.filter = `blur(${this.blurRadius}px)`;
            offCtx.drawImage(this.originalImage, 0, 0, w, h);
            offCtx.filter = 'none';
            this._blurCache = off;
        }
        ctx.drawImage(this._blurCache, 0, 0);
    }

    // ── Player cutout ─────────────────────────────────────────────────────────
    _drawPlayerCutout(ctx, w, h) {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = w;
        maskCanvas.height = h;
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.drawImage(this.maskImage, 0, 0, w, h);
        const maskData = maskCtx.getImageData(0, 0, w, h).data;

        const tmp = document.createElement('canvas');
        tmp.width = w;
        tmp.height = h;
        const tmpCtx = tmp.getContext('2d');
        tmpCtx.drawImage(this.originalImage, 0, 0, w, h);
        const imgData = tmpCtx.getImageData(0, 0, w, h);

        for (let i = 0; i < imgData.data.length; i += 4) {
            imgData.data[i + 3] = maskData[i];
        }
        tmpCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(tmp, 0, 0);
    }

    // ── Selection highlight ───────────────────────────────────────────────────
    _drawSelection(ctx) {
        const el = this.elements.find(e => e.id === this.selectedId);
        if (!el) return;

        let x, y, w, h;
        if (el.type === 'text') {
            ctx.font = `${el.fontWeight || 'bold'} ${el.fontSize}px ${el.fontFamily || 'Impact'}`;
            w = ctx.measureText(el.content).width;
            h = el.fontSize * 1.2;
            x = el.x;
            y = el.y;
        } else if (el.type === 'rect') {
            x = el.x;
            y = el.y;
            w = el.width;
            h = el.height;
        } else if (el.type === 'logo') {
            x = el.x;
            y = el.y;
            w = el.width;
            h = el.height;
        }

        ctx.strokeStyle = '#00cfff';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x - 6, y - 6, w + 12, h + 12);
        ctx.setLineDash([]);
    }

    _updateMetadataUI() {
    const metaPanel = document.getElementById('te-meta-panel');
    if (!metaPanel || !this.metadata) return;

    const m = this.metadata;

    metaPanel.classList.remove('hidden');

    document.getElementById('te-meta-home').textContent =
        m.home_team_short || m.home_team || '—';

    document.getElementById('te-meta-away').textContent =
        m.visiting_team_short || m.visiting_team || '—';

    document.getElementById('te-meta-score').textContent =
        m.score || '—';

    document.getElementById('te-meta-time').textContent =
        m.game_time ? `⏱ ${m.game_time}` : '';

    document.getElementById('te-meta-event').textContent =
        m.event_type ? `· ${m.event_type}` : '';
}

    // =========================================================================
    // Element Management
    // =========================================================================
    addText(content = 'TEXT', x = 80, y = 80, opts = {}) {
        const el = {
            id: ++this._idCounter,
            type: 'text',
            content,
            x,
            y,
            fontSize: opts.fontSize || 85,
            color: opts.color || '#ffffff',
            strokeColor: opts.strokeColor || '#000000',
            strokeWidth: opts.strokeWidth || 8,
            fontFamily: opts.fontFamily || 'Impact',
            fontWeight: opts.fontWeight || 'bold',
            layer: opts.layer || 'background'
        };
        this.elements.push(el);
        this.selectedId = el.id;
        this._render();
        this._emitElements();
        return el.id;
    }

    addRect(x = 50, y = 50, opts = {}) {
        const el = {
            id: ++this._idCounter,
            type: 'rect',
            x,
            y,
            width: opts.width || 410,
            height: opts.height || 100,
            fillColor: opts.fillColor || '#0a1628',
            opacity: opts.opacity ?? 0.85, 
            strokeColor: opts.strokeColor || null,
            strokeWidth: opts.strokeWidth || 0,
            cornerRadius: opts.cornerRadius || 0,
            layer: opts.layer || 'background'
        };
        this.elements.push(el);
        this.selectedId = el.id;
        this._render();
        this._emitElements();
        return el.id;
    }

    async addLogo(imageUrl, x = 50, y = 50, opts = {}) {
        if (!this.loadedLogos[imageUrl]) {
            this.loadedLogos[imageUrl] = await this._loadImg(imageUrl);
        }

        const el = {
            id: ++this._idCounter,
            type: 'logo',
            imageUrl,
            x,
            y,
            width: opts.width || 100,
            height: opts.height || 100,
            layer: opts.layer || 'foreground'
        };
        this.elements.push(el);
        this.selectedId = el.id;
        this._render();
        this._emitElements();
        return el.id;
    }

    // ── Sample the dominant identity colour from a team logo ──────────────────
    // Strategy: convert every opaque pixel to HSL, keep only the most saturated
    // ones (these are the vivid team colours, not white/black/grey), then average
    // their RGB.  Falls back to navy if the logo is monochrome or too small.
    _sampleLogoColor(imageEl) {
        try {
            const SIZE = 64;
            const off = document.createElement('canvas');
            off.width = SIZE; off.height = SIZE;
            const ctx = off.getContext('2d');
            ctx.drawImage(imageEl, 0, 0, SIZE, SIZE);
            const px = ctx.getImageData(0, 0, SIZE, SIZE).data;

            // Collect every opaque pixel as [r, g, b, saturation]
            const colorful = [];
            for (let i = 0; i < px.length; i += 4) {
                const a = px[i + 3];
                if (a < 100) continue; // skip transparent / semi-transparent

                const r = px[i] / 255;
                const g = px[i + 1] / 255;
                const b = px[i + 2] / 255;

                const max = Math.max(r, g, b);
                const min = Math.min(r, g, b);
                const lightness = (max + min) / 2;
                const delta = max - min;

                // Skip near-white (lightness > 0.85) and near-black (lightness < 0.12)
                if (lightness > 0.85 || lightness < 0.12) continue;

                // HSL saturation
                const sat = delta === 0 ? 0 : delta / (1 - Math.abs(2 * lightness - 1));

                // Only keep pixels with meaningful saturation (not grey)
                if (sat < 0.25) continue;

                colorful.push({ r: px[i], g: px[i + 1], b: px[i + 2], sat });
            }

            if (colorful.length < 10) {
                // Logo is mostly monochrome (e.g. black/white badge) — fall back to navy
                return '#0a1628';
            }

            // Sort by saturation descending, take the top 30% most vivid pixels
            colorful.sort((a, b) => b.sat - a.sat);
            const top = colorful.slice(0, Math.max(10, Math.floor(colorful.length * 0.3)));

            // Average the vivid pixels
            const rAvg = Math.round(top.reduce((s, c) => s + c.r, 0) / top.length);
            const gAvg = Math.round(top.reduce((s, c) => s + c.g, 0) / top.length);
            const bAvg = Math.round(top.reduce((s, c) => s + c.b, 0) / top.length);

            // Darken by ~40% so it reads as a background without losing the hue
            const r = Math.round(rAvg * 0.6);
            const g = Math.round(gAvg * 0.6);
            const b = Math.round(bAvg * 0.6);

            return `#${r.toString(16).padStart(2,'0')}${g.toString(16).padStart(2,'0')}${b.toString(16).padStart(2,'0')}`;
        } catch {
            return '#0a1628';
        }
    }

    // ── Add both team logos (TV2 style — side by side at bottom) ──────────────
    async addTeamLogos() {
        await this._ensureMetadataLoaded();

        if (!this.metadata) {
            this.showError('No metadata loaded for this keyframe');
            return;
        }
        const { home_team_logo, visiting_team_logo, home_team_short, visiting_team_short } = this.metadata;
        const w = this.canvas.width;
        const h = this.canvas.height;

        if (!home_team_logo && !visiting_team_logo) {
            this.showError('No team logos found in metadata');
            return;
        }

        // ── Layout constants (shared with addScore) ────────────────────────
        const LOGO_SIZE = 200;
        const PAD       = Math.round(w * 0.03);
        const BOTTOM_Y  = h - PAD - LOGO_SIZE;
        // ───────────────────────────────────────────────────────────────────

        // Home team — bottom-left: navy backing rect + logo on top
        if (home_team_logo) {
            this.addRect(PAD, BOTTOM_Y, {
                width: LOGO_SIZE + 30, height: LOGO_SIZE + 30,
                fillColor: '#0a1628',
                strokeColor: null, strokeWidth: 0,
                cornerRadius: 0,
                layer: 'foreground'
            });
            this.showStatus(`Loading ${home_team_short || 'home'} logo…`);
            await this.addLogo(home_team_logo, PAD + 15, BOTTOM_Y + 15, {
                width: LOGO_SIZE, height: LOGO_SIZE, layer: 'foreground'
            });
        }

        // Away team — bottom-right: navy backing rect + logo on top
        if (visiting_team_logo) {
            const awayX = w - PAD - (LOGO_SIZE + 30);
            this.addRect(awayX, BOTTOM_Y, {
                width: LOGO_SIZE + 30, height: LOGO_SIZE + 30,
                fillColor: '#0a1628',
                strokeColor: null, strokeWidth: 0,
                cornerRadius: 0,
                layer: 'foreground'
            });
            this.showStatus(`Loading ${visiting_team_short || 'away'} logo…`);
            await this.addLogo(visiting_team_logo, awayX + 15, BOTTOM_Y + 15, {
                width: LOGO_SIZE, height: LOGO_SIZE, layer: 'foreground'
            });
        }

        this.showStatus('✓ Team logos added');
    }

    // ── Add score (TV2 style — large numbers beside each logo, bottom) ─────────
    async addScore() {
        await this._ensureMetadataLoaded();

        if (!this.metadata) {
            this.showError('No metadata loaded for this keyframe');
            return;
        }
        const { score, home_team_logo, visiting_team_logo } = this.metadata;

        if (!score) {
            this.showError('No score found in metadata');
            return;
        }

        // Parse "3-0" or "3 - 0"
        const parts      = score.split(/[-–]/).map(s => s.trim());
        const homeScore  = parts[0] ?? '?';
        const awayScore  = parts[1] ?? '?';

        const w = this.canvas.width;
        const h = this.canvas.height;

        // ── Same layout constants as addTeamLogos ──────────────────────────
        const LOGO_SIZE   = 200;
        const PAD         = Math.round(w * 0.03);
        const BOTTOM_Y    = h - PAD - LOGO_SIZE;
        const SCORE_FONT  = 200;
        // Score backing rect is as wide as one large digit + side padding
        const SCORE_RW = 140;
        // ───────────────────────────────────────────────────────────────────

        // Try to get dominant colour from the loaded logo images
        let homeColor = '#0a1628';
        let awayColor = '#0a1628';
        if (home_team_logo) {
            try {
                if (!this.loadedLogos[home_team_logo])
                    this.loadedLogos[home_team_logo] = await this._loadImg(home_team_logo);
                homeColor = this._sampleLogoColor(this.loadedLogos[home_team_logo]);
            } catch { /* keep default */ }
        }
        if (visiting_team_logo) {
            try {
                if (!this.loadedLogos[visiting_team_logo])
                    this.loadedLogos[visiting_team_logo] = await this._loadImg(visiting_team_logo);
                awayColor = this._sampleLogoColor(this.loadedLogos[visiting_team_logo]);
            } catch { /* keep default */ }
        }

        // Home score — immediately right of home logo box
        const homeScoreX = PAD + (LOGO_SIZE + 30);
        this.addRect(homeScoreX, BOTTOM_Y, {
            width: SCORE_RW + 30, height: LOGO_SIZE + 30,
            fillColor: homeColor,
            strokeColor: null, strokeWidth: 0,
            cornerRadius: 0,
            layer: 'foreground'
        });

        const rectW = SCORE_RW + 30;
        const rectH = LOGO_SIZE + 30;

        this.addText(
            homeScore,
            homeScoreX + rectW / 2,
            BOTTOM_Y + rectH / 2,
            {
                fontSize: SCORE_FONT,
                color: '#ffffff',
                strokeColor: '#000000',
                strokeWidth: Math.max(2, Math.round(SCORE_FONT * 0.035)),
                fontFamily: 'Impact',
                fontWeight: 'bold',
                layer: 'foreground'
            }
        );

        // Away score — immediately left of away logo box
        const awayLogoX  = w - PAD - (LOGO_SIZE + 30);
        const awayScoreX = awayLogoX - (SCORE_RW + 30);
        this.addRect(awayScoreX, BOTTOM_Y, {
            width: SCORE_RW + 30, height: LOGO_SIZE + 30,
            fillColor: awayColor,
            strokeColor: null, strokeWidth: 0,
            cornerRadius: 0,
            layer: 'foreground'
        });
        this.addText(
            awayScore,
            awayScoreX + rectW / 2,
            BOTTOM_Y + rectH / 2,
            {
                fontSize: SCORE_FONT,
                color: '#ffffff',
                strokeColor: '#000000',
                strokeWidth: Math.max(2, Math.round(SCORE_FONT * 0.035)),
                fontFamily: 'Impact',
                fontWeight: 'bold',
                layer: 'foreground'
            }
        );

        this.showStatus('✓ Score overlay added');
    }

    updateElement(id, updates) {
        const el = this.elements.find(e => e.id === id);
        if (!el) return;
        Object.assign(el, updates);
        this._render();
        this._emitElements();
    }

    deleteElement(id) {
        this.elements = this.elements.filter(e => e.id !== id);
        if (this.selectedId === id) this.selectedId = null;
        this._render();
        this._emitElements();
    }

    moveElementUp(id) {
        const idx = this.elements.findIndex(e => e.id === id);
        if (idx < this.elements.length - 1) {
            [this.elements[idx], this.elements[idx + 1]] = [this.elements[idx + 1], this.elements[idx]];
            this._render();
            this._emitElements();
        }
    }

    moveElementDown(id) {
        const idx = this.elements.findIndex(e => e.id === id);
        if (idx > 0) {
            [this.elements[idx], this.elements[idx - 1]] = [this.elements[idx - 1], this.elements[idx]];
            this._render();
            this._emitElements();
        }
    }


    // =========================================================================
    // Mouse Interaction
    // =========================================================================
    _canvasXY(e) {
        const r = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - r.left) * (this.canvas.width / r.width),
            y: (e.clientY - r.top) * (this.canvas.height / r.height)
        };
    }

    _hitTest(px, py, el) {
        if (el.type === 'text') {
            this.ctx.font = `${el.fontWeight || 'bold'} ${el.fontSize}px ${el.fontFamily || 'Impact'}`;
            const w = this.ctx.measureText(el.content).width;
            const h = el.fontSize * 1.2;
            return px >= el.x && px <= el.x + w && py >= el.y && py <= el.y + h;
        } else if (el.type === 'rect') {
            return px >= el.x && px <= el.x + el.width && py >= el.y && py <= el.y + el.height;
        } else if (el.type === 'badge') {
            return px >= el.x && px <= el.x + el.width && py >= el.y && py <= el.y + el.height;
        } else if (el.type === 'logo') {
            return px >= el.x && px <= el.x + el.width && py >= el.y && py <= el.y + el.height;
        }
        return false;
    }

    _onMouseDown(e) {
        const { x, y } = this._canvasXY(e);
        for (const el of [...this.elements].reverse()) {
            if (this._hitTest(x, y, el)) {
                this.selectedId = el.id;
                this.isDragging = true;
                this.dragOffset = { x: x - el.x, y: y - el.y };
                this._render();
                this._emitElements();
                return;
            }
        }
        this.selectedId = null;
        this._render();
        this._emitElements();
    }

    _onMouseMove(e) {
        const { x, y } = this._canvasXY(e);
        if (this.isDragging && this.selectedId) {
            const el = this.elements.find(e => e.id === this.selectedId);
            if (el) {
                el.x = x - this.dragOffset.x;
                el.y = y - this.dragOffset.y;
                this._render();
            }
        } else {
            const over = this.elements.some(el => this._hitTest(x, y, el));
            this.canvas.style.cursor = over ? 'move' : 'default';
        }
    }

    _onMouseUp() {
        this.isDragging = false;
    }

    // =========================================================================
    // Download
    // =========================================================================
    async downloadThumbnail() {
        this.showStatus('Rendering…');
        try {
            const maskFilename = this._maskFilename || null;

            const res = await fetch('/api/create-thumbnail', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    keyframe_filename: this.imageFilename,
                    mask_filename: maskFilename,
                    elements: this.elements,
                    background_color: this.backgroundColor,
                    player_layer: this.playerLayer,
                    blur_background: this.blurBackground,
                    blur_radius: this.blurRadius
                })
            });
            const data = await res.json();
            if (!data.success) throw new Error(data.error);

            const a = document.createElement('a');
            a.href = data.thumbnail_url + '?t=' + Date.now();
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
    // Events
    // =========================================================================
    _emitElements() {
        document.dispatchEvent(new CustomEvent('elementsChanged', {
            detail: { elements: this.elements, selectedId: this.selectedId }
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
// UI Integration
// =============================================================================

function openThumbnailEditor(filename, metadata = null) {
    let modal = document.getElementById('te-modal');
    if (!modal) {
        modal = _buildModal();
        document.body.appendChild(modal);
    }

    modal.style.display = 'flex';

    const editor = new ThumbnailEditor(filename, 'te-canvas');
    window._editor = editor;
    //editor.init(metadata);
    editor.init(); // no metadata here
    _wireControls(editor);
}

function closeThumbnailEditor() {
    const m = document.getElementById('te-modal');
    if (m) m.style.display = 'none';
    window._editor = null;
}

// Expose helpers
function teSegment() {
    const p = document.getElementById('te-prompt');
    window._editor.segmentPlayer(p ? p.value : '');
}
function teAddText() {
    const v = prompt('Enter text:', 'HIGHLIGHTS');
    if (v) window._editor.addText(v);
}
function teAddRect() {
    window._editor.addRect(100, 100, { width: 410, height: 100, fillColor: '#0a1628', cornerRadius: 8 });
}
function teDelElement(id) {
    window._editor.deleteElement(id);
}
function teMoveUp(id) {
    window._editor.moveElementUp(id);
}
function teMoveDown(id) {
    window._editor.moveElementDown(id);
}
function teProp(id, k, v) {
    window._editor.updateElement(id, { [k]: v });
}
function teBgColor(v) {
    window._editor.backgroundColor = v;
    window._editor._render();
}
function teDownload() {
    window._editor.downloadThumbnail();
}
function tePlayerLayer(layer) {
    window._editor.playerLayer = layer;
    window._editor._render();
    document.getElementById('te-pl-bg').classList.toggle('active', layer === 'background');
    document.getElementById('te-pl-fg').classList.toggle('active', layer === 'foreground');
}
function teBlur(checked) {
    window._editor.blurBackground = checked;
    window._editor._blurCache = null;
    window._editor._render();
    const row = document.getElementById('te-blur-row');
    if (row) row.classList.toggle('hidden', !checked);
}
function teBlurRadius(v) {
    window._editor.blurRadius = parseInt(v);
    window._editor._blurCache = null;
    window._editor._render();
    const d = document.getElementById('te-blur-val');
    if (d) d.textContent = v + 'px';
}
function teApplyTemplate(name) {
    window._editor.applyTemplate(name);
}
function teAddTeamLogos() {
    window._editor.addTeamLogos();
}
function teAddScore() {
    window._editor.addScore();
}

// ... continue in next message with modal HTML and element list UI

// =============================================================================
// Modal HTML with enhanced controls
// =============================================================================
function _buildModal() {
    const m = document.createElement('div');
    m.id = 'te-modal';
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


          <!-- Add Elements -->
          <section class="control-section">
            <h3>Add Elements</h3>
            <div class="add-buttons">
              <button class="btn-add" onclick="teAddText()">+ Text</button>
              <button class="btn-add" onclick="teAddRect()">+ Rectangle</button>
            </div>
            <div id="te-meta-buttons" class="meta-buttons hidden">
              <button class="btn-add btn-meta" onclick="teAddTeamLogos()">⚽ + Team Logos</button>
              <button class="btn-add btn-meta" onclick="teAddScore()">🏆 + Score</button>
            </div>
          </section>

          <!-- Metadata info panel -->
          <div id="te-meta-panel" class="meta-panel hidden">
            <div class="meta-panel-inner">
              <div class="meta-match">
                <span id="te-meta-home" class="meta-team">—</span>
                <span id="te-meta-score" class="meta-score-badge">— : —</span>
                <span id="te-meta-away" class="meta-team">—</span>
              </div>
              <div class="meta-detail">
                <span id="te-meta-time"></span>
                <span id="te-meta-event"></span>
              </div>
            </div>
          </div>

          <!-- Segmentation -->
          <section class="control-section">
            <h3>Player Segmentation</h3>
            <label class="field-label">Prompt</label>
            <input id="te-prompt" class="prompt-input" type="text"
                   value="soccer player" placeholder="e.g. goalkeeper…" />
            <button class="btn-primary" onclick="teSegment()">Run Segmentation</button>

            <div id="te-layer-row" class="player-layer-row hidden">
              <label class="field-label">Player layer</label>
              <div class="layer-toggle-group">
                <button id="te-pl-bg" class="layer-toggle-btn" onclick="tePlayerLayer('background')">Behind</button>
                <button id="te-pl-fg" class="layer-toggle-btn active" onclick="tePlayerLayer('foreground')">Front</button>
              </div>
            </div>

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

          <!-- Elements List -->
          <section class="control-section">
            <h3>Elements</h3>
            <div id="te-elements"></div>
          </section>

          <!-- Background -->
          <section class="control-section">
            <h3>Background</h3>
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

// =============================================================================
// Wire controls
// =============================================================================
function _wireControls(editor) {
    // Reset event listeners
    document.removeEventListener('elementsChanged', _onElementsChanged);
    document.removeEventListener('editorStatus', _onEditorStatus);
    document.addEventListener('elementsChanged', _onElementsChanged);
    document.addEventListener('editorStatus', _onEditorStatus);

    // Reset UI state
    const layerRow = document.getElementById('te-layer-row');
    if (layerRow) layerRow.classList.add('hidden');

    const blurChk = document.getElementById('te-blur-chk');
    if (blurChk) blurChk.checked = false;

    const blurRow = document.getElementById('te-blur-row');
    if (blurRow) blurRow.classList.add('hidden');

    const elList = document.getElementById('te-elements');
    if (elList) elList.innerHTML = '<p class="empty-text-list">No elements yet</p>';

    //  Metadata UI — always same initial state
    const metaButtons = document.getElementById('te-meta-buttons');
    const metaPanel   = document.getElementById('te-meta-panel');

    // Buttons should ALWAYS be visible (user can click them anytime)
    if (metaButtons) metaButtons.classList.remove('hidden');

    // Info panel should be hidden until metadata is actually loaded
    if (metaPanel) metaPanel.classList.add('hidden');
}


function _onEditorStatus(e) {
    const el = document.getElementById('te-status');
    if (!el) return;
    el.textContent = e.detail.msg;
    el.className = 'status-message ' + e.detail.type;
    clearTimeout(el._t);
    el._t = setTimeout(() => {
        el.textContent = '';
        el.className = 'status-message';
    }, 4000);
}

function _onElementsChanged(e) {
    const list = document.getElementById('te-elements');
    if (!list) return;
    const { elements, selectedId } = e.detail;

    if (!elements.length) {
        list.innerHTML = '<p class="empty-text-list">No elements yet</p>';
        return;
    }

    list.innerHTML = '<p style="margin:0 0 6px;font-size:10px;color:#555;line-height:1.4;">▲ = draw on top &nbsp;·&nbsp; ▼ = draw behind</p>';

    // Reverse so the topmost element in the list is drawn last (on top)
    [...elements].reverse().forEach(el => {
        const div = document.createElement('div');
        div.className = 'element-item' + (el.id === selectedId ? ' selected' : '');

        let title = '';
        let controls = '';

        if (el.type === 'text') {
            title = `📝 ${el.content}`;
            controls = `
                <label>Text <input type="text" value="${el.content.replace(/"/g, '&quot;')}"
                       onchange="teProp(${el.id},'content',this.value)" /></label>
                <label>Size <input type="number" value="${el.fontSize}" min="10" max="500"
                       onchange="teProp(${el.id},'fontSize',+this.value)" /></label>
                <label>Font
                    <select onchange="teProp(${el.id},'fontFamily',this.value)">
                        <option value="Impact" ${el.fontFamily === 'Impact' ? 'selected' : ''}>Impact</option>
                        <option value="Arial Black" ${el.fontFamily === 'Arial Black' ? 'selected' : ''}>Arial Black</option>
                        <option value="Bebas Neue" ${el.fontFamily === 'Bebas Neue' ? 'selected' : ''}>Bebas Neue</option>
                        <option value="Montserrat" ${el.fontFamily === 'Montserrat' ? 'selected' : ''}>Montserrat</option>
                        <option value="Oswald" ${el.fontFamily === 'Oswald' ? 'selected' : ''}>Oswald</option>
                    </select>
                </label>
                <label>Color <input type="color" value="${el.color}"
                       onchange="teProp(${el.id},'color',this.value)" /></label>
                <label>Outline <input type="color" value="${el.strokeColor}"
                       onchange="teProp(${el.id},'strokeColor',this.value)" /></label>
            `;
        } else if (el.type === 'rect') {
            title = `▭ Rectangle`;
            controls = `
                <label>Width <input type="number" value="${el.width}" min="10" max="2000"
                       onchange="teProp(${el.id},'width',+this.value)" /></label>
                <label>Height <input type="number" value="${el.height}" min="10" max="2000"
                       onchange="teProp(${el.id},'height',+this.value)" /></label>
                <label>Fill <input type="color" value="${el.fillColor}"
                       onchange="teProp(${el.id},'fillColor',this.value)" /></label>
                <label>Corner <input type="number" value="${el.cornerRadius}" min="0" max="50"
                       onchange="teProp(${el.id},'cornerRadius',+this.value)" /></label>
                <label>Opacity <input type="range" min="0" max="1" step="0.05" value="${el.opacity ?? 1}"
                        onchange="teProp(${el.id},'opacity',+this.value)" />
                </label>
            `;
        } else if (el.type === 'logo') {
            title = `🖼 Logo`;
            controls = `
                <label>Width <input type="number" value="${el.width}" min="10" max="500"
                       onchange="teProp(${el.id},'width',+this.value)" /></label>
                <label>Height <input type="number" value="${el.height}" min="10" max="500"
                       onchange="teProp(${el.id},'height',+this.value)" /></label>
            `;
        }

        div.innerHTML = `
          <div class="element-header">
            <strong>${title}</strong>
            <div style="display:flex;gap:4px;align-items:center;">
              <button class="btn-delete" title="Move up (draw on top)" style="background:#1a2e1a;color:#8bc98b;font-size:13px;"
                      onclick="teMoveUp(${el.id})">▲</button>
              <button class="btn-delete" title="Move down (draw behind)" style="background:#1a1a2e;color:#8b8bc9;font-size:13px;"
                      onclick="teMoveDown(${el.id})">▼</button>
              <button class="btn-delete" onclick="teDelElement(${el.id})">&times;</button>
            </div>
          </div>
          <div class="element-controls">
            ${controls}
            <label>Layer
              <select onchange="teProp(${el.id},'layer',this.value)">
                <option value="background" ${el.layer === 'background' ? 'selected' : ''}>Behind player</option>
                <option value="foreground" ${el.layer === 'foreground' ? 'selected' : ''}>In front</option>
              </select>
            </label>
          </div>
        `;
        list.appendChild(div);
    });
}