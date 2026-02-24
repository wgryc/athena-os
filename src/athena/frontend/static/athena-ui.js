/**
 * Athena UI Utilities
 *
 * Themed replacements for native alert(), confirm(), and prompt() dialogs.
 * Each utility creates its modal DOM lazily and reuses the existing Athena
 * modal CSS classes (.modal-overlay, .modal-panel, etc.).
 *
 * Usage:
 *   await athenaAlert.show({ title: 'Error', message: 'Something went wrong' });
 *   const ok = await athenaConfirm.show({ title: 'Delete', message: 'Are you sure?' });
 *   const values = await athenaPrompt.show({ title: 'Add Widget', inputs: [...] });
 */

/* ===== athenaAlert ===== */
const athenaAlert = {
    overlay: null,
    titleEl: null,
    messageEl: null,
    confirmBtn: null,
    resolvePromise: null,
    handleKeydown: null,

    createModal() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'modal-overlay athena-ui-modal';

        const panel = document.createElement('div');
        panel.className = 'modal-panel athena-ui-panel';

        // Header
        const header = document.createElement('div');
        header.className = 'modal-header';
        this.titleEl = document.createElement('h2');
        header.appendChild(this.titleEl);

        // Body
        const body = document.createElement('div');
        body.className = 'modal-body';
        this.messageEl = document.createElement('p');
        this.messageEl.className = 'athena-ui-message';
        body.appendChild(this.messageEl);

        // Footer
        const footer = document.createElement('div');
        footer.className = 'modal-footer';
        this.confirmBtn = document.createElement('button');
        this.confirmBtn.className = 'btn-primary';
        this.confirmBtn.textContent = 'OK';
        this.confirmBtn.addEventListener('click', () => this.confirm());
        footer.appendChild(this.confirmBtn);

        panel.appendChild(header);
        panel.appendChild(body);
        panel.appendChild(footer);
        this.overlay.appendChild(panel);

        this.handleKeydown = (e) => {
            if (this.overlay.style.display !== 'flex') return;
            if (e.key === 'Escape' || e.key === 'Enter') {
                e.preventDefault();
                this.confirm();
            }
        };
        document.addEventListener('keydown', this.handleKeydown);

        document.body.appendChild(this.overlay);
    },

    show({ title = 'Alert', message = '' } = {}) {
        if (!this.overlay) this.createModal();
        this.titleEl.textContent = title;
        this.messageEl.textContent = message;
        this.overlay.style.display = 'flex';
        setTimeout(() => this.confirmBtn.focus(), 50);
        return new Promise(resolve => { this.resolvePromise = resolve; });
    },

    confirm() {
        this.overlay.style.display = 'none';
        if (this.resolvePromise) this.resolvePromise();
    },
};

/* ===== athenaConfirm ===== */
const athenaConfirm = {
    overlay: null,
    titleEl: null,
    messageEl: null,
    confirmBtn: null,
    cancelBtn: null,
    resolvePromise: null,
    handleKeydown: null,

    createModal() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'modal-overlay athena-ui-modal';

        const panel = document.createElement('div');
        panel.className = 'modal-panel athena-ui-panel';

        // Header
        const header = document.createElement('div');
        header.className = 'modal-header';
        this.titleEl = document.createElement('h2');
        header.appendChild(this.titleEl);

        // Body
        const body = document.createElement('div');
        body.className = 'modal-body';
        this.messageEl = document.createElement('p');
        this.messageEl.className = 'athena-ui-message';
        body.appendChild(this.messageEl);

        // Footer
        const footer = document.createElement('div');
        footer.className = 'modal-footer';

        this.cancelBtn = document.createElement('button');
        this.cancelBtn.className = 'btn-secondary';
        this.cancelBtn.textContent = 'Cancel';
        this.cancelBtn.addEventListener('click', () => this.cancel());

        this.confirmBtn = document.createElement('button');
        this.confirmBtn.className = 'btn-primary';
        this.confirmBtn.textContent = 'OK';
        this.confirmBtn.addEventListener('click', () => this.confirm());

        footer.appendChild(this.cancelBtn);
        footer.appendChild(this.confirmBtn);

        panel.appendChild(header);
        panel.appendChild(body);
        panel.appendChild(footer);
        this.overlay.appendChild(panel);

        this.handleKeydown = (e) => {
            if (this.overlay.style.display !== 'flex') return;
            if (e.key === 'Escape') {
                e.preventDefault();
                this.cancel();
            } else if (e.key === 'Enter') {
                e.preventDefault();
                this.confirm();
            }
        };
        document.addEventListener('keydown', this.handleKeydown);

        document.body.appendChild(this.overlay);
    },

    show({ title = 'Confirm', message = '' } = {}) {
        if (!this.overlay) this.createModal();
        this.titleEl.textContent = title;
        this.messageEl.textContent = message;
        this.overlay.style.display = 'flex';
        setTimeout(() => this.confirmBtn.focus(), 50);
        return new Promise(resolve => { this.resolvePromise = resolve; });
    },

    confirm() {
        this.overlay.style.display = 'none';
        if (this.resolvePromise) this.resolvePromise(true);
    },

    cancel() {
        this.overlay.style.display = 'none';
        if (this.resolvePromise) this.resolvePromise(false);
    },
};

/* ===== athenaPrompt ===== */
const athenaPrompt = {
    overlay: null,
    titleEl: null,
    inputsContainer: null,
    confirmBtn: null,
    cancelBtn: null,
    inputs: [],
    resolvePromise: null,
    handleKeydown: null,

    createModal() {
        this.overlay = document.createElement('div');
        this.overlay.className = 'modal-overlay athena-ui-modal';

        const panel = document.createElement('div');
        panel.className = 'modal-panel athena-ui-panel';

        // Header
        const header = document.createElement('div');
        header.className = 'modal-header';
        this.titleEl = document.createElement('h2');
        header.appendChild(this.titleEl);

        // Body
        const body = document.createElement('div');
        body.className = 'modal-body';
        this.inputsContainer = document.createElement('div');
        this.inputsContainer.className = 'athena-ui-inputs';
        body.appendChild(this.inputsContainer);

        // Footer
        const footer = document.createElement('div');
        footer.className = 'modal-footer';

        this.cancelBtn = document.createElement('button');
        this.cancelBtn.className = 'btn-secondary';
        this.cancelBtn.textContent = 'Cancel';
        this.cancelBtn.addEventListener('click', () => this.cancel());

        this.confirmBtn = document.createElement('button');
        this.confirmBtn.className = 'btn-primary';
        this.confirmBtn.textContent = 'OK';
        this.confirmBtn.addEventListener('click', () => this.confirm());

        footer.appendChild(this.cancelBtn);
        footer.appendChild(this.confirmBtn);

        panel.appendChild(header);
        panel.appendChild(body);
        panel.appendChild(footer);
        this.overlay.appendChild(panel);

        this.handleKeydown = (e) => {
            if (this.overlay.style.display !== 'flex') return;
            if (e.key === 'Escape') {
                e.preventDefault();
                this.cancel();
            } else if (e.key === 'Enter' && this.inputs.length === 1) {
                e.preventDefault();
                this.confirm();
            }
        };
        document.addEventListener('keydown', this.handleKeydown);

        document.body.appendChild(this.overlay);
    },

    /**
     * Show the prompt modal.
     *
     * @param {Object} config
     * @param {string} config.title - Modal title
     * @param {Array}  config.inputs - Array of input descriptors:
     *   { name: string, label?: string, type?: string, value?: string, description?: string }
     * @returns {Promise<Object|null>} Object keyed by input name, or null if cancelled.
     */
    show({ title = 'Enter Values', inputs = [] } = {}) {
        if (!this.overlay) this.createModal();

        this.titleEl.textContent = title;
        this.inputsContainer.innerHTML = '';
        this.inputs = [];

        inputs.forEach(cfg => {
            const group = document.createElement('div');
            group.className = 'athena-ui-input-group';

            if (cfg.label) {
                const label = document.createElement('label');
                label.className = 'athena-ui-label';
                label.textContent = cfg.label;
                if (cfg.description) {
                    const hint = document.createElement('span');
                    hint.className = 'athena-ui-hint';
                    hint.textContent = cfg.description;
                    label.appendChild(hint);
                }
                group.appendChild(label);
            }

            const input = document.createElement('input');
            input.type = cfg.type || 'text';
            input.value = cfg.value || '';
            input.name = cfg.name || '';
            input.className = 'athena-ui-input';
            if (cfg.placeholder) input.placeholder = cfg.placeholder;

            group.appendChild(input);
            this.inputsContainer.appendChild(group);
            this.inputs.push(input);
        });

        this.overlay.style.display = 'flex';
        setTimeout(() => {
            if (this.inputs.length > 0) this.inputs[0].focus();
        }, 50);

        return new Promise(resolve => { this.resolvePromise = resolve; });
    },

    confirm() {
        const result = {};
        this.inputs.forEach(input => {
            result[input.name] = input.value;
        });
        this.overlay.style.display = 'none';
        if (this.resolvePromise) this.resolvePromise(result);
    },

    cancel() {
        this.overlay.style.display = 'none';
        if (this.resolvePromise) this.resolvePromise(null);
    },
};
