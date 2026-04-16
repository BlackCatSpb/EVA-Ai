/* CogniFlex Web GUI — app.js */
(function () {
    'use strict';

    /* ── Simple $ helper (replaces jQuery) ── */
    function $(selector) {
        if (selector.startsWith('.')) {
            return document.getElementsByClassName(selector.slice(1))[0];
        }
        return document.getElementById(selector.slice(1));
    }

    /* ── $$ helper for multiple elements (like querySelectorAll) ── */
    function $$(selector) {
        return document.querySelectorAll(selector);
    }

    /* ── State ── */
    let token = null;
    let userId = null;
    let sessions = [];
    let activeSessionId = null;
    let sidebarOpen = false;
    let settingsState = {
        auto_learn: true,
        memory_enabled: true,
        sre_enabled: true
    };

    /* ── SSE Event Source ── */
    let eventSource = null;
    
    /* ── Message Storage for Action Buttons ── */
    const messageStore = {};
    let selectedContext = null;

    function initSSE() {
        if (eventSource) {
            eventSource.close();
        }
        eventSource = new EventSource('/api/events/stream');

        eventSource.addEventListener('pipeline.start', function(e) {
            showGenerationProgress('start');
        });

        eventSource.addEventListener('pipeline.complete', function(e) {
            setTimeout(hideGenerationProgress, 300);
        });

        eventSource.addEventListener('pipeline.failed', function(e) {
            setTimeout(hideGenerationProgress, 500);
        });

        eventSource.onerror = function() {
            if (eventSource.readyState === EventSource.CLOSED) return;
            setTimeout(initSSE, 5000);
        };
    }

    function showGenerationProgress(phase) {
        // Progress bar removed - typing indicator shown via updateMessageText
    }

    function hideGenerationProgress() {
        // Progress bar removed
    }

    function updateGenerationStep(step) {
        const label = $('#genLabel', genProgressEl);
        const fill = $('#genFill', genProgressEl);
        const stepA = $('#stepA', genProgressEl);
        const stepB = $('#stepB', genProgressEl);
        const stepC = $('#stepC', genProgressEl);

        switch(step) {
            case 'model_a':
                if (label) label.textContent = 'Model A...';
                if (fill) fill.style.width = '25%';
                if (stepA) { stepA.className = 'gen-step active'; }
                break;
            case 'model_a_done':
                if (stepA) { stepA.className = 'gen-step done'; }
                if (stepB) { stepB.className = 'gen-step active'; }
                if (label) label.textContent = 'Model B...';
                if (fill) fill.style.width = '55%';
                break;
            case 'model_b':
                if (label) label.textContent = 'Model B...';
                if (fill) fill.style.width = '55%';
                if (stepB) { stepB.className = 'gen-step active'; }
                break;
            case 'model_b_done':
                if (stepB) { stepB.className = 'gen-step done'; }
                if (stepC) { stepC.className = 'gen-step active'; }
                if (label) label.textContent = 'Finalizing...';
                if (fill) fill.style.width = '85%';
                break;
            case 'complete':
                if (stepC) { stepC.className = 'gen-step done'; }
                if (label) label.textContent = 'Done!';
                if (fill) fill.style.width = '100%';
                stopGenTimer();
                break;
            case 'failed':
                if (label) label.textContent = 'Error';
                if (fill) { fill.style.width = '100%'; fill.style.background = '#ef4444'; }
                stopGenTimer();
                break;
        }
    }

    function startGenTimer() {
        stopGenTimer();
        genStartTime = Date.now();
        genTimerInterval = setInterval(function() {
            const elapsed = ((Date.now() - genStartTime) / 1000).toFixed(1);
            const timeEl = $('#genTime');
            if (timeEl) timeEl.textContent = elapsed + 'с';

            const elapsedSec = Date.now() - genStartTime;
            if (elapsedSec > 30000 && genProgressEl) {
                let notice = genProgressEl.querySelector('.gen-timeout-notice');
                if (!notice) {
                    notice = document.createElement('div');
                    notice.className = 'gen-timeout-notice';
                    notice.innerHTML = '<span class="timeout-spinner"></span>Генерация занимает больше времени. Ожидание...';
                    genProgressEl.appendChild(notice);
                }
            }
        }, 200);
    }

    function stopGenTimer() {
        if (genTimerInterval) {
            clearInterval(genTimerInterval);
            genTimerInterval = null;
        }
    }

    function hideGenerationProgress() {
        if (genProgressEl) {
            genProgressEl.remove();
            genProgressEl = null;
        }
        stopGenTimer();
    }

    /* ── API ── */
    async function api(path, opts = {}) {
        console.log('API call:', path, opts);
        const headers = { 'Content-Type': 'application/json' };
        if (userId) headers['X-User-ID'] = userId;
        
        let url = `/api${path}`;
        if (opts.params) {
            const params = new URLSearchParams(opts.params);
            url += '?' + params.toString();
            delete opts.params;
        }
        
        let r;
        try {
            console.log('Fetching:', url, 'body:', opts.body);
            r = await fetch(url, {
                ...opts,
                headers: { ...headers, ...opts.headers },
                body: opts.body ? JSON.stringify(opts.body) : undefined
            });
            console.log('Response:', r.status, r.statusText);
        } catch (e) {
            console.error('Network error:', e);
            toast('Ошибка сети. Проверьте подключение.', 'error');
            throw e;
        }

        if (!r.ok) {
            console.error('HTTP error:', r.status);
            if (r.status === 403) {
                toast('Доступ запрещён. Сессия истекла.', 'error');
            } else if (r.status === 404) {
                toast('Ресурс не найден.', 'error');
            } else if (r.status >= 500) {
                toast('Ошибка сервера. Попробуйте позже.', 'error');
            } else {
                toast("Ошибка " + r.status, 'error');
            }
            try {
                const errData = await r.json();
                if (errData.error) throw new Error(errData.error);
            } catch {
                throw new Error(`HTTP ${r.status}`);
            }
        }

        return r.json();
    }

    /* ── Toast ── */
    const activeToasts = [];
    function toast(msg, type = 'info') {
        const c = $('#toastContainer');
        const t = document.createElement('div');
        t.className = `toast ${type}`;
        t.textContent = msg;
        c.appendChild(t);
        activeToasts.push(t);
        setTimeout(() => {
            t.style.animation = 'toastOut .2s ease forwards';
            setTimeout(() => {
                t.remove();
                const idx = activeToasts.indexOf(t);
                if (idx > -1) activeToasts.splice(idx, 1);
            }, 200);
        }, 5000);
    }

    /* ── Login ── */
    $('#loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const u = $('#loginUsername').value.trim();
        const p = $('#loginPassword').value.trim();
        if (!u || !p) return;

        console.log('Login: Submitting', u);
        $('#loginError').textContent = 'Проверка...';
        
        try {
            console.log('Login: Calling API');
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: u, password: p })
            });
            console.log('Login: Response status', response.status);
            
            const d = await response.json();
            console.log('Login: Response data', d);
            
            if (d.error || !response.ok) {
                $('#loginError').textContent = d.error || 'Ошибка входа';
                console.error('Login error:', d.error);
                return;
            }
            
            console.log('Login: Success, switching screens');
            token = true;
            sessions = d.sessions || [];
            activeSessionId = d.session_id;
            // Extract userId from first session or generate from username
            userId = sessions[0]?.user_id || sessions[0]?.id || d.user;
            $('#loginScreen').style.display = 'none';
            $('#appContainer').style.display = 'flex';
            $('#userName').textContent = d.user;
            $('#settingsUser').textContent = d.user;
            $('#settingsSession').textContent = d.session_id || '—';
            renderSessions();
            
            // Load chat history for active session after login
            if (activeSessionId) {
                loadSessionMessages(activeSessionId);
            } else if (sessions.length > 0) {
                // If no active session but have sessions, load the most recent one
                activeSessionId = sessions[0].id;
                loadSessionMessages(activeSessionId);
            } else {
                showWelcome();
            }
            
            loadSettings();
            loadInitialData();
            toast('Добро пожаловать', 'success');
        } catch(err) {
            console.error('Login exception:', err);
            $('#loginError').textContent = 'Ошибка подключения: ' + err.message;
        }
    });

    /* ── Logout ── */
    $('#logoutBtn').addEventListener('click', () => {
        token = null;
        userId = null;
        sessions = [];
        activeSessionId = null;
        $('#appContainer').style.display = 'none';
        $('#loginScreen').style.display = 'flex';
        $('#loginUsername').value = '';
        $('#loginPassword').value = '';
        $('#chatMessages').innerHTML = '';
    });

    /* ── Sessions ── */
    function renderSessions() {
        const c = $('#sessionList');
        c.innerHTML = '';
        sessions.forEach(s => {
            const el = document.createElement('div');
            el.className = `session-item${s.id === activeSessionId ? ' active' : ''}`;
            el.dataset.id = s.id;

            const createdDate = s.created_at ? formatDate(s.created_at) : '';
            const msgCount = s.message_count !== undefined ? s.message_count : (s.context ? s.context.length : 0);

            el.innerHTML = `
                <div class="session-top">
                    <span class="session-name">${esc(s.name || 'Без названия')}</span>
                    <button class="session-del" data-id="${s.id}" title="Удалить">×</button>
                </div>
                ${createdDate ? `<div class="session-meta">${createdDate}${msgCount ? ' · ' + msgCount + ' сообщ.' : ''}</div>` : ''}
            `;

            el.addEventListener('click', (e) => {
                if (e.target.classList.contains('session-del')) return;
                selectSession(s.id);
            });

            el.addEventListener('dblclick', (e) => {
                if (e.target.classList.contains('session-del')) return;
                startRenameSession(s.id, el);
            });

            el.querySelector('.session-del').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteSession(s.id);
            });

            c.appendChild(el);
        });
    }

    function formatDate(dateStr) {
        try {
            const d = new Date(dateStr);
            const now = new Date();
            const diff = now - d;
            if (diff < 60000) return 'Только что';
            if (diff < 3600000) return Math.floor(diff / 60000) + ' мин. назад';
            if (diff < 86400000) return Math.floor(diff / 3600000) + ' ч. назад';
            return d.toLocaleDateString('ru-RU', { day: 'numeric', month: 'short' });
        } catch {
            return '';
        }
    }

    function startRenameSession(id, el) {
        const nameSpan = el.querySelector('.session-name');
        const currentName = nameSpan.textContent;
        const input = document.createElement('input');
        input.className = 'session-rename-input';
        input.value = currentName;
        nameSpan.replaceWith(input);
        input.focus();
        input.select();

        const finishRename = async () => {
            const newName = input.value.trim();
            if (newName && newName !== currentName) {
                try {
                    await api(`/session/${id}`, {
                        method: 'PUT',
                        body: { name: newName }
                    });
                    const s = sessions.find(s => s.id === id);
                    if (s) s.name = newName;
                    toast('Сессия переименована', 'success');
                } catch {
                    toast('Ошибка переименования', 'error');
                }
            }
            renderSessions();
        };

        input.addEventListener('blur', finishRename);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') input.blur();
            if (e.key === 'Escape') {
                input.value = currentName;
                input.blur();
            }
        });
    }

    async function selectSession(id) {
        activeSessionId = id;
        renderSessions();
        await loadSessionMessages(id);
    }

    async function deleteSession(id) {
        try {
            await api(`/session/${id}`, { method: 'DELETE' });
            sessions = sessions.filter(s => s.id !== id);
            if (activeSessionId === id) {
                activeSessionId = sessions[0]?.id || null;
            }
            renderSessions();
            if (activeSessionId) {
                await loadSessionMessages(activeSessionId);
            } else {
                $('#chatMessages').innerHTML = '';
                showWelcome();
            }
            toast('Сессия удалена', 'success');
        } catch {
            toast('Ошибка удаления сессии', 'error');
        }
    }

    $('#newChatBtn').addEventListener('click', async () => {
        const d = await api('/sessions', { method: 'POST', body: { name: 'Новый чат' } });
        sessions = d.sessions || [];
        activeSessionId = d.session_id;
        renderSessions();
        $('#chatMessages').innerHTML = '';
        showWelcome();
    });

    /* ── Messages ── */
    function showWelcome() {
        const c = $('#chatMessages');
        if (c.children.length === 0) {
            c.innerHTML = `<div class="welcome"><h2>EVA</h2><p>Напишите сообщение, чтобы начать</p></div>`;
        }
    }

    async function loadSessionMessages(id) {
        try {
            const d = await api(`/session/${id}`);
            const c = $('#chatMessages');
            c.innerHTML = '';
            
            // First try chat_history, then fall back to context
            const chatHistory = d.chat_history || d.context || [];
            
            if (chatHistory.length === 0) {
                showWelcome();
                return;
            }
            
            // chat_history has {role: 'user'/'assistant', content: '...'}
            chatHistory.forEach(msg => {
                if (msg.role === 'user' || msg.role === 'user_message') {
                    addMsg('user', msg.content, msg.entities);
                } else if (msg.role === 'assistant' || msg.role === 'assistant_message') {
                    addMsg('system', msg.content);
                }
            });
        } catch { /* ignore */ }
    }

    function addMsg(role, text, entities, reasoning, fileData, msgId) {
        const c = $('#chatMessages');
        const welcome = c.querySelector('.welcome');
        if (welcome) welcome.remove();

        // Generate message ID if not provided
        const messageId = msgId || ('msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9));

        const div = document.createElement('div');
        div.className = 'msg';
        div.id = messageId;

        let reasoningHtml = '';
        let fileHtml = '';
        
        // Show file if attached
        if (fileData && fileData.filename) {
            fileHtml = `
                <div class="msg-file">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
                    <span>${esc(fileData.filename)}</span>
                </div>
            `;
        }
        
        // Use reasoning from brain if available
        if (reasoning) {
            // Check if reasoning is structured (array of steps) or plain text
            if (Array.isArray(reasoning) && reasoning.length > 0) {
                // Render structured reasoning steps
                const stepsHtml = reasoning.map((step, idx) => {
                    const phase = step.phase || step.action || 'unknown';
                    const thought = step.thought || step.text || '';
                    const conf = step.confidence || 0;
                    const confClass = conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low';
                    const confLabel = (conf * 100).toFixed(0) + '%';
                    const icons = {
                        'generation': '💭',
                        'model_a_generation': '🧠',
                        'model_b_generation': '💡',
                        'query_analysis': '🔍',
                        'model_selection': '⚙️',
                        'context_retrieval': '📚',
                        'condensed': '📝',
                        'extended': '📖',
                        'quality_check': '✅',
                        'quality_check_a': '✅',
                        'quality_check_b': '✅',
                        'final_synthesis': '📝',
                        'contradiction_check': '⚖️',
                        'ethics_check': '🛡️',
                        'web_search': '🌐',
                        'refinement': '✨',
                        'self_dialog': '💬'
                    };
                    const icon = step.icon || icons[phase] || '🔹';
                    return `
                        <div class="reasoning-step">
                            <span class="step-icon">${icon}</span>
                            <span class="step-num">${idx + 1}</span>
                            <span class="step-phase">${esc(phase)}</span>
                            <span class="step-thought">${esc(thought)}</span>
                            <span class="step-conf ${confClass}">${confLabel}</span>
                        </div>
                    `;
                }).join('');
                reasoningHtml = `
                    <div class="msg-reasoning collapsed" id="reasoning-${messageId}">
                        <button class="reasoning-toggle" onclick="toggleReasoning(this, '${messageId}')">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                            Рассуждения (${reasoning.length} шагов)
                        </button>
                        <div class="reasoning-body">
                            <div class="reasoning-steps">${stepsHtml}</div>
                        </div>
                        <span class="reasoning-collapsed-badge" onclick="toggleReasoning(this.parentElement.querySelector('.reasoning-toggle'), '${messageId}')">
                            🧠 ${reasoning.length} шагов рассуждений — показать
                        </span>
                    </div>
                `;
            } else {
                // Plain text reasoning
                const reasoningText = typeof reasoning === 'string' ? reasoning : JSON.stringify(reasoning, null, 2);
                reasoningHtml = `
                    <div class="msg-reasoning collapsed" id="reasoning-${messageId}">
                        <button class="reasoning-toggle" onclick="toggleReasoning(this, '${messageId}')">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                            Рассуждения
                        </button>
                        <div class="reasoning-body"><pre>${esc(reasoningText)}</pre></div>
                        <span class="reasoning-collapsed-badge" onclick="toggleReasoning(this.parentElement.querySelector('.reasoning-toggle'), '${messageId}')">
                            🧠 Показать рассуждения
                        </span>
                    </div>
                `;
            }
        } else if (entities && entities.length > 0) {
            // Fallback to entity-based reasoning
            const steps = entities.map((e, i) =>
                `${i + 1}. [${e.type}] ${e.keyword}`
            ).join('\n');
            reasoningHtml = `
                <div class="msg-reasoning collapsed" id="reasoning-${messageId}">
                    <button class="reasoning-toggle" onclick="toggleReasoning(this, '${messageId}')">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                        Рассуждения
                    </button>
                    <div class="reasoning-body"><pre>${esc(steps)}</pre></div>
                    <span class="reasoning-collapsed-badge" onclick="toggleReasoning(this.parentElement.querySelector('.reasoning-toggle'), '${messageId}')">
                        🧠 Показать рассуждения
                    </span>
                </div>
            `;
        }

        const roleLabel = role === 'user' ? 'Вы' : 'EVA';
        const roleClass = role;

        // Add action buttons only for assistant messages
        const actionsHtml = role !== 'user' ? `
            <div class="msg-actions">
                <button class="msg-action-btn copy-btn" data-msg-id="${messageId}" title="Копировать">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
                    Копировать
                </button>
                <button class="msg-action-btn like" data-msg-id="${messageId}" data-rating="1" title="Полезно">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
                    Полезно
                </button>
                <button class="msg-action-btn dislike" data-msg-id="${messageId}" data-rating="-1" title="Неверно">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>
                    Неверно
                </button>
                <button class="msg-action-btn regenerate" data-msg-id="${messageId}" title="Перегенерировать">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/></svg>
                    Перегенерировать
                </button>
            </div>
        ` : '';

        div.innerHTML = `
            <div class="msg-inner">
                <div class="msg-role ${roleClass}">${roleLabel}</div>
                <div class="msg-text" id="text-${messageId}">${formatText(text)}</div>
                ${fileHtml}
                ${reasoningHtml}
                ${actionsHtml}
            </div>
        `;
        c.appendChild(div);
        c.scrollTop = c.scrollHeight;
        
        // Store message text for action buttons
        messageStore[messageId] = {
            text: text,
            role: role,
            reasoning: reasoning
        };
        
        // Add event listeners for action buttons
        const copyBtn = div.querySelector('.copy-btn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const data = messageStore[messageId];
                if (data && data.text) {
                    navigator.clipboard.writeText(data.text).then(() => {
                        toast('Скопировано', 'success');
                    }).catch(() => {
                        toast('Ошибка копирования', 'error');
                    });
                }
            });
        }
        
        const likeBtn = div.querySelector('.like');
        if (likeBtn) {
            likeBtn.addEventListener('click', () => {
                const data = messageStore[messageId];
                if (data) {
                    rateMessage(data.text, 1);
                }
            });
        }
        
        const dislikeBtn = div.querySelector('.dislike');
        if (dislikeBtn) {
            dislikeBtn.addEventListener('click', () => {
                const data = messageStore[messageId];
                if (data) {
                    rateMessage(data.text, -1);
                }
            });
        }
        
        const regenerateBtn = div.querySelector('.regenerate');
        if (regenerateBtn) {
            regenerateBtn.addEventListener('click', () => {
                regenerateMessage(div);
            });
        }
        
        // Apply syntax highlighting to code blocks
        if (window.hljs) {
            div.querySelectorAll('pre code').forEach((block) => {
                hljs.highlightElement(block);
            });
        }
    }

    /**
     * Toggle reasoning panel open/closed (Grok-like behavior)
     */
    function toggleReasoning(button, messageId) {
        const container = document.getElementById('reasoning-' + messageId);
        if (!container) return;
        
        const toggle = container.querySelector('.reasoning-toggle');
        const body = container.querySelector('.reasoning-body');
        
        const isOpen = toggle.classList.contains('open');
        
        if (isOpen) {
            toggle.classList.remove('open');
            container.classList.add('collapsed');
        } else {
            toggle.classList.add('open');
            container.classList.remove('collapsed');
            // Auto-scroll to bottom of reasoning content
            if (body) {
                setTimeout(() => { body.scrollTop = body.scrollHeight; }, 50);
            }
        }
    }
    
    // Make toggleReasoning globally accessible
    window.toggleReasoning = toggleReasoning;

    /**
     * Update live reasoning panel during generation (streaming)
     */
    function updateLiveReasoning(msgId, steps) {
        const msgEl = document.getElementById(msgId);
        if (!msgEl) return;
        
        let reasoningContainer = msgEl.querySelector('.msg-reasoning');
        
        if (!reasoningContainer) {
            // Create reasoning container for the first time
            reasoningContainer = document.createElement('div');
            reasoningContainer.className = 'msg-reasoning collapsed';
            reasoningContainer.id = 'reasoning-' + msgId;
            reasoningContainer.innerHTML = `
                <button class="reasoning-toggle" onclick="toggleReasoning(this, '${msgId}')">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                    <span class="reasoning-toggle-text">Рассуждения...</span>
                </button>
                <div class="reasoning-body">
                    <div class="reasoning-steps"></div>
                </div>
                <span class="reasoning-collapsed-badge" onclick="toggleReasoning(this.parentElement.querySelector('.reasoning-toggle'), '${msgId}')">
                    🧠 Показать рассуждения
                </span>
            `;
            
            // Insert before actions
            const actions = msgEl.querySelector('.msg-actions');
            if (actions) {
                msgEl.querySelector('.msg-inner').insertBefore(reasoningContainer, actions);
            } else {
                msgEl.querySelector('.msg-inner').appendChild(reasoningContainer);
            }
        }
        
        // Update steps
        const stepsContainer = reasoningContainer.querySelector('.reasoning-steps');
        if (stepsContainer) {
            const icons = {
                'generation': '💭', 'model_a_generation': '🧠', 'model_b_generation': '💡',
                'query_analysis': '🔍', 'model_selection': '⚙️', 'context_retrieval': '📚',
                'condensed': '📝', 'extended': '📖', 'quality_check': '✅',
                'final_synthesis': '📝', 'contradiction_check': '⚖️', 'ethics_check': '🛡️',
                'web_search': '🌐', 'refinement': '✨', 'self_dialog': '💬'
            };
            
            stepsContainer.innerHTML = steps.map((step, idx) => {
                const phase = step.phase || step.action || 'unknown';
                const thought = step.thought || step.text || '';
                const conf = step.confidence || 0;
                const confClass = conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low';
                const confLabel = (conf * 100).toFixed(0) + '%';
                const icon = step.icon || icons[phase] || '🔹';
                return `
                    <div class="reasoning-step">
                        <span class="step-icon">${icon}</span>
                        <span class="step-num">${idx + 1}</span>
                        <span class="step-phase">${esc(phase)}</span>
                        <span class="step-thought">${esc(thought)}</span>
                        <span class="step-conf ${confClass}">${confLabel}</span>
                    </div>
                `;
            }).join('');
            
            // Update toggle text
            const toggleText = reasoningContainer.querySelector('.reasoning-toggle-text');
            if (toggleText) toggleText.textContent = `Рассуждения (${steps.length} шагов)`;
            
            // Update badge
            const badge = reasoningContainer.querySelector('.reasoning-collapsed-badge');
            if (badge) badge.innerHTML = `🧠 ${steps.length} шагов рассуждений — показать`;
            
            // Auto-scroll to bottom of reasoning if open
            const body = reasoningContainer.querySelector('.reasoning-body');
            if (body && reasoningContainer.querySelector('.reasoning-toggle.open')) {
                body.scrollTop = body.scrollHeight;
            }
        }
    }
    
    /**
     * Add reasoning to an existing message (after generation completes)
     */
    function addReasoningToMessage(msgId, reasoning) {
        const msgEl = document.getElementById(msgId);
        if (!msgEl || !reasoning || reasoning.length === 0) return;
        
        // Remove any existing reasoning
        const existing = msgEl.querySelector('.msg-reasoning');
        if (existing) existing.remove();
        
        // Create new reasoning with the final data
        const reasoningContainer = document.createElement('div');
        reasoningContainer.className = 'msg-reasoning collapsed';
        reasoningContainer.id = 'reasoning-' + msgId;
        
        const icons = {
            'generation': '💭', 'model_a_generation': '🧠', 'model_b_generation': '💡',
            'query_analysis': '🔍', 'model_selection': '⚙️', 'context_retrieval': '📚',
            'condensed': '📝', 'extended': '📖', 'quality_check': '✅',
            'final_synthesis': '📝', 'contradiction_check': '⚖️', 'ethics_check': '🛡️',
            'web_search': '🌐', 'refinement': '✨', 'self_dialog': '💬'
        };
        
        const stepsHtml = reasoning.map((step, idx) => {
            const phase = step.phase || step.action || 'unknown';
            const thought = step.thought || step.text || '';
            const conf = step.confidence || 0;
            const confClass = conf >= 0.7 ? 'high' : conf >= 0.4 ? 'medium' : 'low';
            const confLabel = (conf * 100).toFixed(0) + '%';
            const icon = step.icon || icons[phase] || '🔹';
            return `
                <div class="reasoning-step">
                    <span class="step-icon">${icon}</span>
                    <span class="step-num">${idx + 1}</span>
                    <span class="step-phase">${esc(phase)}</span>
                    <span class="step-thought">${esc(thought)}</span>
                    <span class="step-conf ${confClass}">${confLabel}</span>
                </div>
            `;
        }).join('');
        
        reasoningContainer.innerHTML = `
            <button class="reasoning-toggle" onclick="toggleReasoning(this, '${msgId}')">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
                Рассуждения (${reasoning.length} шагов)
            </button>
            <div class="reasoning-body">
                <div class="reasoning-steps">${stepsHtml}</div>
            </div>
            <span class="reasoning-collapsed-badge" onclick="toggleReasoning(this.parentElement.querySelector('.reasoning-toggle'), '${msgId}')">
                🧠 ${reasoning.length} шагов рассуждений — показать
            </span>
        `;
        
        // Insert before actions
        const actions = msgEl.querySelector('.msg-actions');
        if (actions) {
            msgEl.querySelector('.msg-inner').insertBefore(reasoningContainer, actions);
        } else {
            msgEl.querySelector('.msg-inner').appendChild(reasoningContainer);
        }
    }
    
    /**
     * Auto-collapse all reasoning panels for a message after generation completes
     */
    function collapseAllReasoning(msgId) {
        const msgEl = document.getElementById(msgId);
        if (!msgEl) return;
        
        const reasoningContainer = msgEl.querySelector('.msg-reasoning');
        if (reasoningContainer) {
            reasoningContainer.classList.add('collapsed');
            const toggle = reasoningContainer.querySelector('.reasoning-toggle');
            if (toggle) toggle.classList.remove('open');
        }
    }

    function addTyping() {
        // Typing indicator removed - status shown inside message
    }

    function removeTyping() {
        // Typing indicator removed
    }
    
    let genStartTime = null;
    let genTimerInterval = null;
    
    function startGenTimer() {
        genStartTime = Date.now();
        genTimerInterval = setInterval(() => {
            const elapsed = ((Date.now() - genStartTime) / 1000).toFixed(1);
            const genStatus = document.querySelector('.gen-status');
            if (genStatus) {
                genStatus.textContent = `Генерирую... ${elapsed}с`;
            }
        }, 100);
    }
    
    function stopGenTimer() {
        if (genTimerInterval) {
            clearInterval(genTimerInterval);
            genTimerInterval = null;
        }
    }

    /* ── File Upload ── */
    let currentFile = null;
    let currentFileData = null;

    $('#attachBtn').addEventListener('click', () => {
        $('#fileInput').click();
    });

    $('#fileInput').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        currentFile = file;
        $('#fileName').textContent = file.name;
        $('#filePreview').style.display = 'flex';

        // Upload file to server
        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await resp.json();
            if (data.error) {
                toast(data.error, 'error');
                clearFile();
                return;
            }
            currentFileData = data;
            toast(`Файл "${file.name}" загружен`, 'success');
            
            // Показать превью извлечённого текста
            if (data.extracted_text && data.extracted_text.trim()) {
                const preview = $('#extractedTextPreview');
                const text = data.extracted_text.trim();
                const shortText = text.length > 300 ? text.substring(0, 300) + '...' : text;
                preview.innerHTML = `<div class="extracted-label">📄 Распознанный текст:</div><div class="extracted-content">${esc(shortText)}</div>`;
                preview.style.display = 'block';
            } else {
                const preview = $('#extractedTextPreview');
                preview.style.display = 'none';
            }
        } catch (err) {
            toast('Ошибка загрузки файла', 'error');
            clearFile();
        }
    });

    $('#removeFile').addEventListener('click', clearFile);

    function clearFile() {
        currentFile = null;
        currentFileData = null;
        $('#fileInput').value = '';
        $('#filePreview').style.display = 'none';
        const preview = $('#extractedTextPreview');
        if (preview) {
            preview.style.display = 'none';
            preview.innerHTML = '';
        }
    }

    /* ── Documents ── */
    function loadDocuments() {
        if (!activeSessionId) return;
        
        // Загружаем обычные документы
        api('/documents', { params: { session_id: activeSessionId } }).then(data => {
            if (data.error) return;
            
            const docs = data.documents || [];
            
            // Также загружаем документы из DocumentVirtualMemory
            api('/documents/memory', { params: { session_id: activeSessionId } }).then(memData => {
                const memDocs = memData.documents || {};
                const docEl = $('#documentList');
                
                if (docEl) {
                    let html = '';
                    
                    // Обычные документы
                    if (docs.length > 0) {
                        html += docs.map(d => `
                            <div class="doc-item">
                                <div class="doc-icon">📄</div>
                                <div class="doc-info">
                                    <div class="doc-name">${esc(d.filename || 'Unknown')}</div>
                                    <div class="doc-meta">${d.doc_type || 'file'} · ${esc(d.file_id?.substring(0, 8) || '')}</div>
                                </div>
                            </div>
                        `).join('');
                    }
                    
                    // Документы в виртуальной памяти
                    const memDocEntries = Object.entries(memDocs);
                    if (memDocEntries.length > 0) {
                        html += '<div class="doc-section-title">📚 В виртуальной памяти:</div>';
                        html += memDocEntries.map(([docId, d]) => {
                            const stats = d.stats || {};
                            const totalPages = stats.total_pages || '?';
                            return `
                            <div class="doc-item doc-item-memory">
                                <div class="doc-icon">📖</div>
                                <div class="doc-info">
                                    <div class="doc-name">${esc(d.title || 'Unknown')}</div>
                                    <div class="doc-meta">${totalPages} стр. · ID: ${esc(docId.substring(0, 8))}</div>
                                </div>
                            </div>
                        `}).join('');
                    }
                    
                    if (html === '') {
                        html = '<div class="empty-state">Нет загруженных документов</div>';
                    }
                    
                    docEl.innerHTML = html;
                }
            }).catch(() => {
                // Fallback: показываем только обычные документы
                if (docEl) {
                    if (docs.length > 0) {
                        docEl.innerHTML = docs.map(d => `
                            <div class="doc-item">
                                <div class="doc-icon">📄</div>
                                <div class="doc-info">
                                    <div class="doc-name">${esc(d.filename || 'Unknown')}</div>
                                    <div class="doc-meta">${d.doc_type || 'file'} · ${esc(d.file_id?.substring(0, 8) || '')}</div>
                                </div>
                            </div>
                        `).join('');
                    } else {
                        docEl.innerHTML = '<div class="empty-state">Нет загруженных документов</div>';
                    }
                }
            });
        }).catch(() => {});
    }
    
    /* ── Knowledge Graph ── */
    function loadKnowledgeGraph() {
        api('/knowledge-graph', { params: { action: 'get' } }).then(data => {
            if (data.error) return;
            
            const nodes = data.nodes || [];
            const kgEl = $('#knowledgeGraph');
            
            if (kgEl) {
                if (nodes.length > 0) {
                    kgEl.innerHTML = nodes.map(n => `
                        <div class="kg-node">
                            <div class="kg-node-name">${esc(n.name || n.id)}</div>
                            <div class="kg-node-content">${esc(n.content || '')}</div>
                        </div>
                    `).join('');
                } else {
                    kgEl.innerHTML = '<div class="empty-state">Граф знаний пуст</div>';
                }
            }
        }).catch(() => {});
    }
    
    /* ── Memory ── */
    function loadMemory() {
        api('/memory-graph').then(data => {
            if (data.error) {
                toast('Ошибка загрузки памяти', 'error');
                return;
            }
            const stats = data.stats || {};
            $('#totalNodes').textContent = stats.total_nodes || data.nodes?.length || 0;
            $('#totalEdges').textContent = stats.total_edges || data.edges?.length || 0;
            
            // Simple list view of nodes
            const graphEl = $('#memoryGraph');
            if (data.nodes && data.nodes.length > 0) {
                graphEl.innerHTML = data.nodes.slice(0, 50).map(n => 
                    `<div class="node-item">${esc(n.label || n.id)}</div>`
                ).join('');
            } else {
                graphEl.innerHTML = '<div class="empty-state">Нет данных о памяти</div>';
            }
        }).catch(() => {
            toast('Ошибка загрузки памяти', 'error');
        });
    }
    
    $('#refreshMemory')?.addEventListener('click', () => {
        loadMemory();
        loadDocuments();
        loadKnowledgeGraph();
    });

    /* ── Chat ── */
    
    // Streaming mode - включен по умолчанию для плавного отображения
    let streamingMode = true; // localStorage.getItem('streamingMode') === 'true';
    
    function showContextIndicator(text) {
        let indicator = $('#contextIndicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'contextIndicator';
            indicator.className = 'context-indicator';
            const inputBox = $('.chat-input-box');
            if (inputBox) {
                inputBox.insertBefore(indicator, inputBox.firstChild);
            }
        }
        const shortText = text.length > 100 ? text.substring(0, 100) + '...' : text;
        indicator.innerHTML = `
            <span class="context-label">Контекст:</span>
            <span class="context-text">${esc(shortText)}</span>
            <button class="context-clear" onclick="clearSelectedContext()" title="Очистить контекст">×</button>
        `;
        indicator.style.display = 'flex';
    }
    
    function clearSelectedContext() {
        selectedContext = null;
        const indicator = $('#contextIndicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }
    
    // Make it global for onclick handler
    window.clearSelectedContext = clearSelectedContext;
    
    function buildMessageWithContext(userText) {
        if (selectedContext) {
            const contextMessage = `Контекст из предыдущего сообщения:\n"""${selectedContext}"""\n\nВопрос пользователя: ${userText}`;
            return contextMessage;
        }
        return userText;
    }
    
    function sendMessage() {
        // Always use streaming mode
        sendMessageStreaming();
    }
    
    function sendMessageStreaming() {
        // Отправка сообщения с streaming ответом (SSE)
        const input = $('#chatInput');
        const text = input.value.trim();
        if ((!text && !currentFileData) || !activeSessionId) return;

        // Build message with context if selected
        const messageWithContext = buildMessageWithContext(text);
        
        let msgText = text;
        if (currentFileData) {
            msgText = text || `Проанализируй файл ${currentFileData.filename}`;
        }

        input.value = '';
        input.style.height = 'auto';
        
        // Clear context after sending
        if (selectedContext) {
            clearSelectedContext();
        }
        
        addMsg('user', msgText, null, null, currentFileData);
        
        // Создаем сообщение для ассистента сразу (будем обновлять)
        const msgId = 'msg-' + Date.now();
        addMsg('system', '', null, null, null, msgId);
        
        // Запускаем таймер генерации
        startGenTimer();
        
        // Live reasoning container (shown during generation)
        let liveReasoningId = 'live-reasoning-' + msgId;
        let reasoningSteps = [];
        
        const body = {
            message: messageWithContext || `Проанализируй файл ${currentFileData?.filename || ''}`,
            session_id: activeSessionId,
            user_id: userId,
            mode: 'extended'
        };
        if (currentFileData) {
            body.file_data = currentFileData;
        }
        
        // Используем XHR с streaming для POST запроса (EventSource не поддерживает POST)
        const xhr = new XMLHttpRequest();
        let fullText = '';
        let buffer = '';
        
        xhr.open('POST', '/api/chat/stream', true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        
        xhr.onprogress = function() {
            const newData = xhr.responseText.substring(buffer.length);
            buffer = xhr.responseText;
            
            // Парсим SSE события
            const lines = newData.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.substring(6));
                        
                        if (data.type === 'chunk') {
                            fullText += data.text;
                            updateMessageText(msgId, fullText, false, data.elapsed_ms);
                        } else if (data.type === 'reasoning_step') {
                            // Live reasoning step during generation
                            reasoningSteps.push(data.step);
                            updateLiveReasoning(msgId, reasoningSteps);
                        } else if (data.type === 'complete') {
                            // complete содержит полный очищенный текст - используем напрямую
                            fullText = data.text;
                            updateMessageText(msgId, fullText, false, data.elapsed_ms);
                            
                            // If reasoning came with complete, add it to the message
                            if (data.reasoning && data.reasoning.length > 0) {
                                addReasoningToMessage(msgId, data.reasoning);
                            }
                        } else if (data.type === 'done') {
                            updateMessageText(msgId, fullText, true);
                            
                            // Auto-collapse reasoning after generation completes
                            collapseAllReasoning(msgId);
                            
                            stopGenTimer();
                            clearFile();
                        } else if (data.type === 'error') {
                            updateMessageText(msgId, 'Ошибка: ' + data.text, true);
                            stopGenTimer();
                            clearFile();
                        }
                    } catch (e) {
                        // Игнорируем неполные JSON
                    }
                }
            }
        };
        
        xhr.onload = function() {
            if (xhr.status !== 200) {
                console.error('XHR Status:', xhr.status);
            }
            stopGenTimer();
            clearFile();
        };
        
        xhr.onerror = function(e) {
            stopGenTimer();
            console.error('XHR Error:', e);
            updateMessageText(msgId, 'Ошибка соединения', true);
            clearFile();
        };
        
        xhr.send(JSON.stringify(body));
    }
    
    function updateMessageText(msgId, text, isComplete = false, elapsedMs = null) {
        // Обновить текст сообщения с форматированием
        const msgEl = document.getElementById(msgId);
        if (msgEl) {
            const contentEl = msgEl.querySelector('.msg-inner .msg-text');
            if (contentEl) {
                // Всегда используем innerHTML с formatText для корректного отображения
                contentEl.innerHTML = formatText(text);
                
                // Подсветка кода при завершении
                if (isComplete && window.hljs) {
                    contentEl.querySelectorAll('pre code').forEach((block) => {
                        hljs.highlightElement(block);
                    });
                }
                
                // Обновляем статус генерации
                if (!isComplete) {
                    let statusEl = msgEl.querySelector('.gen-status');
                    if (!statusEl) {
                        statusEl = document.createElement('div');
                        statusEl.className = 'gen-status';
                        const innerEl = msgEl.querySelector('.msg-inner');
                        if (innerEl) innerEl.appendChild(statusEl);
                    }
                    if (elapsedMs !== null) {
                        statusEl.textContent = `Генерирую... ${(elapsedMs / 1000).toFixed(1)}с`;
                    }
                } else {
                    // Убираем статус при завершении
                    const statusEl = msgEl.querySelector('.gen-status');
                    if (statusEl) statusEl.remove();
                }
                
                // Автопрокрутка
                const chatContainer = $('#chatMessages');
                if (chatContainer) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            }
        }
    }
    
    function updateGenerationProgress(tokens, elapsedMs) {
        // Обновить индикатор генерации
        const progressEl = document.querySelector('.gen-progress-text');
        if (progressEl) {
            progressEl.textContent = `${tokens} токенов, ${(elapsedMs/1000).toFixed(1)}s`;
        }
    }
    
    function sendMessageStandard() {
        // Стандартная отправка сообщения (без streaming)
        const input = $('#chatInput');
        const text = input.value.trim();
        if ((!text && !currentFileData) || !activeSessionId) return;

        // Build message with context if selected
        const messageWithContext = buildMessageWithContext(text);

        let msgText = text;
        if (currentFileData) {
            msgText = text || `Проанализируй файл ${currentFileData.filename}`;
        }

        input.value = '';
        input.style.height = 'auto';
        
        // Clear context after sending
        if (selectedContext) {
            clearSelectedContext();
        }
        
        addMsg('user', msgText, null, null, currentFileData);
        
        addTyping();
        const body = { message: messageWithContext || `Проанализируй файл ${currentFileData?.filename || ''}`, session_id: activeSessionId, user_id: userId };
        if (currentFileData) {
            body.file_data = currentFileData;
        }

        api('/chat', {
            method: 'POST',
            body: body
        }).then(d => {
            hideGenerationProgress();
            removeTyping();
            
            let reasoningData = null;

            if (d.reasoning_steps && d.reasoning_steps.length > 0) {
                reasoningData = d.reasoning_steps;
            }

            if (d.web_search_info) {
                if (!reasoningData) reasoningData = [];
                reasoningData.push({
                    step: reasoningData.length + 1,
                    phase: 'web_search',
                    thought: d.web_search_info,
                    confidence: 0.8
                });
            }

            if (d.self_dialog) {
                const sd = d.self_dialog;
                let sdText = 'Самодиалог: ' + (sd.topic || '');
                if (sd.outcome) sdText += ' → ' + sd.outcome;
                if (sd.gaps && sd.gaps.length > 0) sdText += '. Пробелы: ' + sd.gaps.join(', ');
                if (!reasoningData) reasoningData = [];
                reasoningData.push({
                    step: reasoningData.length + 1,
                    phase: 'self_dialog',
                    thought: sdText,
                    confidence: sd.confidence || 0.5
                });
            }

            if (!reasoningData && d.reasoning) {
                reasoningData = d.reasoning;
            }

            addMsg('system', d.response || 'Нет ответа', null, reasoningData);
            
            if (d.clarification_question) {
                const clarHtml = `
                    <div class="clarification-box">
                        <div class="clarification-icon">❓</div>
                        <div class="clarification-text">${esc(d.clarification_question)}</div>
                    </div>
                `;
                const chatContainer = document.getElementById('chatMessages');
            const lastMsg = chatContainer ? chatContainer.lastElementChild : null;
                const lastMsgInner = lastMsg?.querySelector?.('.msg-inner');
                if (lastMsgInner) {
                    lastMsgInner.insertAdjacentHTML('beforeend', clarHtml);
                }
            }
            
            clearFile();
        }).catch(() => {
            hideGenerationProgress();
            removeTyping();
            addMsg('system', 'Ошибка соединения');
            clearFile();
        });
    }

    $('#sendBtn').addEventListener('click', sendMessage);
    $('#chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    $('#chatInput').addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    /* ── Sidebar toggle ── */
    function createSidebarToggle() {
        const btn = document.createElement('button');
        btn.className = 'sidebar-toggle';
        btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/></svg>`;
        btn.addEventListener('click', toggleSidebar);
        $('.main').prepend(btn);
    }

    function toggleSidebar() {
        sidebarOpen = !sidebarOpen;
        $('#sidebar').classList.toggle('collapsed', !sidebarOpen);
    }

    // Start with sidebar collapsed
    $('#sidebar').classList.add('collapsed');
    createSidebarToggle();

    // Click on sidebar area to also allow collapsing
    // Add a close button to sidebar
    const sidebarCloseBtn = document.createElement('button');
    sidebarCloseBtn.className = 'new-chat-btn';
    sidebarCloseBtn.style.marginBottom = '0';
    sidebarCloseBtn.style.border = 'none';
    sidebarCloseBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg><span>Свернуть</span>`;
    sidebarCloseBtn.addEventListener('click', toggleSidebar);
    $('#sidebar').insertBefore(sidebarCloseBtn, $('#sidebar').firstChild);

    /* ── Navigation ── */
    $$('.nav-item').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.nav-item').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const view = btn.dataset.view;
            $$('.view').forEach(v => v.style.display = 'none');
            $(`#${view}View`).style.display = 'flex';
            
            if (view === 'analytics') {
                loadAnalytics();
            } else if (view === 'wikipedia' || view === 'search') {
                loadWikipedia();
            } else if (view === 'settings') {
                loadSettings();
                $('#settingsSession').textContent = activeSessionId || '—';
            } else if (view === 'monitor') {
                initMonitor();
            }
        });
    });
    
    // Load initial data after login
    function loadInitialData() {
        loadAnalytics();
        loadKnowledge();
    }
    
    // Update analytics every second
    setInterval(() => {
        if ($('#analyticsView') && $('#analyticsView').style.display !== 'none') {
            loadAnalytics();
        }
    }, 1000);

    /* ── Helpers ── */
    function esc(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    /* ── Selection Popup handled below ── */

    // Hide popup on click elsewhere
    document.addEventListener('mousedown', function(e) {
        if (selectionPopup && !selectionPopup.contains(e.target)) {
            selectionPopup.remove();
            selectionPopup = null;
        }
    });

    function formatText(text) {
        if (!text) return '';
        let html = text;
        
        // Normalize Windows line endings
        html = html.replace(/\r\n/g, '\n');
        
        // Escape HTML first (but preserve markdown)
        html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        
        // Handle escape sequences for markdown characters
        html = html.replace(/\\([\\`*_\[\]()#+\-.!|])/g, '$1');
        
        // Math symbols conversion - comprehensive LaTeX support
        const mathSymbols = {
            // Greek letters
            'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε', 'varepsilon': 'ε',
            'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ',
            'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π', 'varpi': 'ϖ',
            'rho': 'ρ', 'varrho': 'ϱ', 'sigma': 'σ', 'varsigma': 'ς', 'tau': 'τ', 'upsilon': 'υ',
            'phi': 'φ', 'varphi': 'ϕ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
            'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ', 'Pi': 'Π',
            'Sigma': 'Σ', 'Upsilon': 'Υ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω',
            // Math symbols
            'infinity': '∞', 'inf': '∞', 'sqrt': '√', 'sum': '∑', 'prod': '∏',
            'integral': '∫', 'partial': '∂', 'nabla': '∇', 'times': '×', 'div': '÷',
            'pm': '±', 'mp': '∓', 'cdot': '·', 'star': '⋆', 'ast': '∗',
            'leq': '≤', 'geq': '≥', 'neq': '≠', 'equiv': '≡', 'approx': '≈',
            'sim': '∼', 'simeq': '≃', 'cong': '≅', 'propto': '∝',
            'forall': '∀', 'exists': '∃', 'nexists': '∄', 'in': '∈', 'notin': '∉',
            'subset': '⊂', 'subseteq': '⊆', 'supset': '⊃', 'supseteq': '⊇',
            'cup': '∪', 'cap': '∩', 'setminus': '∖', 'emptyset': '∅',
            'rightarrow': '→', 'leftarrow': '←', 'Rightarrow': '⇒', 'Leftarrow': '⇐',
            'leftrightarrow': '↔', 'Leftrightarrow': '⇔', 'mapsto': '↦',
            'uparrow': '↑', 'downarrow': '↓', 'Uparrow': '⇑', 'Downarrow': '⇓',
            'angle': '∠', 'perp': '⊥', 'parallel': '∥', 'nparallel': '∦',
            'ldots': '…', 'cdots': '⋯', 'vdots': '⋮', 'ddots': '⋱',
            'infty': '∞', 'aleph': 'ℵ', 'hbar': 'ℏ', 'ell': 'ℓ', 'Re': 'ℜ', 'Im': 'ℑ',
            'wp': '℘', 'prime': '′', 'backprime': '‵', 'surd': '√',
            // Operators
            'cdot': '·', 'bullet': '•', 'circ': '∘', 'odot': '⊙', 'ominus': '⊖',
            'oplus': '⊕', 'otimes': '⊗', 'oslash': '⊘', 'bigcirc': '○',
            'dagger': '†', 'ddagger': '‡', 'amalg': '⨿', 'wr': '≀',
            // Arrows
            'to': '→', 'gets': '←', 'leadsto': '⇝', 'longrightarrow': '⟶',
            'longleftarrow': '⟵', 'Longrightarrow': '⟹', 'Longleftarrow': '⟸'
        };
        
        // LaTeX superscripts/subscripts mapping
        const superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾', 'n': 'ⁿ', 'i': 'ⁱ'
        };
        const subscripts = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
            '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎'
        };
        
        // Handle bullet points with emoji (like 🔹, 🔸) BEFORE bold/italic
        // This handles lines like "🔹 **Разговорный** — текст"
        html = html.replace(/^[🔹🔸•○◆▪][\s]+(\*\*[^*]+\*\*)[—\-:]?\s*(.+)$/gm, '<li><strong>$1</strong> $2</li>');
        
        // Headers
        html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        
        // Bold and italic
        html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
        html = html.replace(/___(.+?)___/g, '<strong><em>$1</em></strong>');
        html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
        html = html.replace(/_(.+?)_/g, '<em>$1</em>');
        
        // Strikethrough
        html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');
        
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Markdown links [text](url)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="md-link">$1</a>');
        
        // Auto-link plain URLs (but not inside code blocks or links already)
        // Use a temporary placeholder for code blocks
        const codeBlockPlaceholders = [];
        html = html.replace(/<code[^>]*>[\s\S]*?<\/code>/g, (match) => {
            codeBlockPlaceholders.push(match);
            return `%%CODEBLOCK_${codeBlockPlaceholders.length - 1}%%`;
        });
        html = html.replace(/<a[^>]*>[\s\S]*?<\/a>/g, (match) => {
            codeBlockPlaceholders.push(match);
            return `%%LINK_${codeBlockPlaceholders.length - 1}%%`;
        });
        // Now convert plain URLs to links
        html = html.replace(/(?<!["'([>])((https?:\/\/)[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" class="md-link">$1</a>');
        // Restore code blocks and links
        codeBlockPlaceholders.forEach((content, idx) => {
            html = html.replace(`%%CODEBLOCK_${idx}%%`, content);
            html = html.replace(`%%LINK_${idx}%%`, content);
        });
        
        // Convert image URLs to images (only direct links to images)
        html = html.replace(/<a href="([^"]+\.(jpg|jpeg|png|gif|webp|bmp))"[^>]*class="md-link"[^>]*>/g, (match, url) => {
            return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="md-link md-image-link" onclick="event.stopPropagation(); event.preventDefault(); this.classList.toggle('expanded');">`;
        });
        
        // Code blocks - handle Windows \r\n, various formats
        html = html.replace(/```(\w*)\r?\n?([\s\S]*?)```/g, function(match, lang, code) {
            const langLabel = lang || 'text';
            const displayLang = langLabel === 'text' ? '' : langLabel;
            const trimmedCode = code.trim();
            return `<pre class="code-block"><div class="code-header"><span class="code-lang">${displayLang}</span><button class="code-copy" onclick="copyCodeBlock(this)">Копировать</button></div><code class="language-${langLabel}">${trimmedCode}</code></pre>`;
        });
        
        // Unordered lists (including emoji bullets like 🔹, 🔸, •, ○)
        html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^[🔹🔸•○] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
        
        // Ordered lists
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        
        // Blockquotes
        html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
        
        // Horizontal rule - более гибкий паттерн
        html = html.replace(/^[\s]*---[\s]*$/gm, '<hr>');
        html = html.replace(/\n---[\s]*\n/g, '\n<hr>\n');
        html = html.replace(/---/g, '<hr>');
        
        // Checkboxes / Task lists (unchecked and checked) - handle inline and multiline
        html = html.replace(/^- \[ \] (.+)$/gm, '<div class="checkbox unchecked"><span class="checkbox-box"></span>$1</div>');
        html = html.replace(/^- \[x\] (.+)$/gm, '<div class="checkbox checked"><span class="checkbox-box checked"></span>$1</div>');
        html = html.replace(/^\* \[ \] (.+)$/gm, '<div class="checkbox unchecked"><span class="checkbox-box"></span>$1</div>');
        html = html.replace(/^\* \[x\] (.+)$/gm, '<div class="checkbox checked"><span class="checkbox-box checked"></span>$1</div>');
        
        // Better handling of multiple newlines - create paragraphs
        html = html.replace(/\n{3,}/g, '\n\n');
        
        // Tables - convert | col1 | col2 | to HTML table
        const tableRegex = /^\|(.+)\|\s*\n\|[-:\s|]+\|\s*\n((?:\|.+\|\s*\n?)+)/gm;
        html = html.replace(tableRegex, function(match, headerRow, bodyRows) {
            const headers = headerRow.split('|').filter(h => h.trim()).map(h => h.trim());
            const rows = bodyRows.trim().split('\n').map(row => 
                row.split('|').filter(c => c.trim()).map(c => c.trim())
            );
            let tableHtml = '<div class="table-wrapper"><table class="md-table"><thead><tr>';
            headers.forEach(h => { tableHtml += `<th>${h}</th>`; });
            tableHtml += '</tr></thead><tbody>';
            rows.forEach(row => {
                tableHtml += '<tr>';
                row.forEach(cell => { tableHtml += `<td>${cell}</td>`; });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody></table></div>';
            return tableHtml;
        });
        
        // Handle superscripts like x^2, x^{10}
        html = html.replace(/(\w)\^\{(\d+)\}/g, (match, base, exp) => {
            const sup = exp.split('').map(c => superscripts[c] || c).join('');
            return base + sup;
        });
        html = html.replace(/(\w)\^(\d)/g, (match, base, exp) => {
            return base + (superscripts[exp] || exp);
        });
        
        // Handle subscripts like x_2, x_{10}
        html = html.replace(/(\w)_\{(\d+)\}/g, (match, base, sub) => {
            const subscript = sub.split('').map(c => subscripts[c] || c).join('');
            return base + subscript;
        });
        html = html.replace(/(\w)_(\d)/g, (match, base, sub) => {
            return base + (subscripts[sub] || sub);
        });
        
        // Math blocks: $$...$$ for display math
        html = html.replace(/\$\$([^$]+)\$\$/g, '<div class="math-block">$$$1$$</div>');
        
        // Math inline: $...$ for inline math
        html = html.replace(/\$([^$\n]+)\$/g, '<span class="math-inline">$$1$</span>');
        
        // Line breaks
        html = html.replace(/\n\n/g, '</p><p>');
        html = html.replace(/\n/g, '<br>');
        
        // Wrap in paragraphs
        html = '<p>' + html + '</p>';
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[123]>)/g, '$1');
        html = html.replace(/(<\/h[123]>)<\/p>/g, '$1');
        html = html.replace(/<p>(<pre>)/g, '$1');
        html = html.replace(/(<\/pre>)<\/p>/g, '$1');
        html = html.replace(/<p>(<ul>)/g, '$1');
        html = html.replace(/(<\/ul>)<\/p>/g, '$1');
        html = html.replace(/<p>(<blockquote>)/g, '$1');
        html = html.replace(/(<\/blockquote>)<\/p>/g, '$1');
        html = html.replace(/<p>(<hr>)<\/p>/g, '$1');
        
        // Apply math symbols conversion (only outside tags)
        // First handle LaTeX \commands
        html = html.replace(/\\([a-zA-Z]+)/g, (match, cmd) => {
            return mathSymbols[cmd] || match;
        });
        
        // Then handle text-based symbols
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        const applyMathSymbols = (node) => {
            if (node.nodeType === Node.TEXT_NODE) {
                let text = node.textContent;
                // Replace operators like <=, >=
                text = text.replace(/<=/g, '≤').replace(/>=/g, '≥');
                text = text.replace(/!=/g, '≠').replace(/=>/g, '⇒');
                text = text.replace(/->/g, '→').replace(/<-/g, '←');
                node.textContent = text;
            } else if (node.nodeType === Node.ELEMENT_NODE && !['code', 'pre', 'script', 'style'].includes(node.tagName.toLowerCase())) {
                Array.from(node.childNodes).forEach(applyMathSymbols);
            }
        };
        Array.from(tempDiv.childNodes).forEach(applyMathSymbols);
        html = tempDiv.innerHTML;
        
        return html;
    }

    /* ── Status Indicator ── */
    const statusEl = $('#systemStatus');
    const statusDot = statusEl?.querySelector('.status-dot');
    const statusText = statusEl?.querySelector('.status-text');

    function updateStatus(status) {
        if (!statusEl) return;
        statusEl.className = 'status-indicator ' + status.state;
        if (statusText) statusText.textContent = status.label;
    }

    async function checkStatus() {
        try {
            const d = await api('/status');
            if (d.status === 'not_initialized') {
                updateStatus({ state: 'offline', label: 'Не подключено' });
            } else if (d.brain_connected && d.brain_running) {
                updateStatus({ state: 'online', label: 'Онлайн' });
            } else if (d.brain_connected && !d.brain_running) {
                updateStatus({ state: 'starting', label: 'Запуск...' });
            } else {
                updateStatus({ state: 'offline', label: 'Офлайн' });
            }
        } catch {
            updateStatus({ state: 'error', label: 'Ошибка' });
        }
    }

    // Poll status every 5 seconds
    checkStatus();
    setInterval(checkStatus, 5000);

    // Init SSE for real-time generation progress
    initSSE();

    /* ── Selection Popup Menu (QWEN-style) ── */
    const selectionPopup = document.createElement('div');
    selectionPopup.className = 'selection-popup';
    selectionPopup.innerHTML = `
        <button class="popup-btn" data-action="copy" title="Копировать">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            <span>Копировать</span>
        </button>
        <div class="popup-divider"></div>
        <button class="popup-btn" data-action="ask" title="Спросить о выделенном">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            <span>Спросить</span>
        </button>
        <button class="popup-btn" data-action="explain" title="Объяснить">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
            <span>Объяснить</span>
        </button>
        <button class="popup-btn" data-action="websearch" title="Найди в интернете">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            <span>Найди в интернете</span>
        </button>
        <button class="popup-btn" data-action="translate" title="Перевести">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 8l6 6"/><path d="M4 14l6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="M22 22l-5-10-5 10"/><path d="M14 18h6"/></svg>
            <span>Перевести</span>
        </button>
        <button class="popup-btn" data-action="rewrite" title="Переписать">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
            <span>Переписать</span>
        </button>
    `;
    selectionPopup.style.display = 'none';
    document.body.appendChild(selectionPopup);

    function showSelectionPopup(x, y) {
        const selection = window.getSelection();
        if (!selection || selection.toString().trim().length === 0) {
            selectionPopup.style.display = 'none';
            return;
        }
        
        selectionPopup.style.display = 'flex';
        selectionPopup.style.left = x + 'px';
        selectionPopup.style.top = y + 'px';
        
        // Adjust if off-screen
        const rect = selectionPopup.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            selectionPopup.style.left = (window.innerWidth - rect.width - 10) + 'px';
        }
        if (rect.bottom > window.innerHeight) {
            selectionPopup.style.top = (y - rect.height - 10) + 'px';
        }
    }

    function hideSelectionPopup() {
        selectionPopup.style.display = 'none';
    }

    document.addEventListener('mouseup', (e) => {
        const selection = window.getSelection();
        if (selection && selection.toString().trim().length > 0) {
            showSelectionPopup(e.pageX, e.pageY);
        } else {
            // Check if click is on popup
            if (!selectionPopup.contains(e.target)) {
                hideSelectionPopup();
            }
        }
    });

    document.addEventListener('mousedown', (e) => {
        if (!selectionPopup.contains(e.target)) {
            hideSelectionPopup();
        }
    });

    selectionPopup.querySelectorAll('.popup-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const selection = window.getSelection();
            const selectedText = selection ? selection.toString().trim() : '';
            if (!selectedText) {
                hideSelectionPopup();
                return;
            }

            const action = btn.dataset.action;
            
            switch (action) {
                case 'copy':
                    navigator.clipboard.writeText(selectedText).then(() => {
                        toast('Скопировано', 'success');
                    }).catch(() => {
                        toast('Ошибка копирования', 'error');
                    });
                    break;
                    
                case 'ask':
                    selectedContext = selectedText;
                    $('#chatInput').value = `По поводу выделенного текста: `;
                    $('#chatInput').focus();
                    // Show context indicator
                    showContextIndicator(selectedText);
                    break;
                    
                case 'explain':
                    $('#chatInput').value = `Объясни что значит: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
                
                case 'websearch':
                    $('#chatInput').value = `Найди информацию в интернете: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
                
                case 'translate':
                    $('#chatInput').value = `Переведи на русский: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
                    
                case 'rewrite':
                    $('#chatInput').value = `Перепиши это более понятно: "${selectedText}"`;
                    $('#chatInput').focus();
                    break;
            }
            
            hideSelectionPopup();
        });
    });

    // Also hide on scroll
    document.addEventListener('scroll', hideSelectionPopup, true);

    /* ── Analytics ── */
    function loadAnalytics() {
        Promise.all([api('/analytics'), api('/metrics')]).then(([analytics, metrics]) => {
            if (analytics.error) {
                toast('Ошибка загрузки аналитики', 'error');
                console.error('Analytics error:', analytics.error);
                return;
            }
            
            // Basic analytics
            $('#analyticsQueries').textContent = analytics.queries || 0;
            $('#analyticsAvgTime').textContent = (analytics.avg_time || 0).toFixed(0) + 'ms';
            $('#analyticsSuccess').textContent = ((analytics.success_rate || 0) * 100).toFixed(0) + '%';
            $('#analyticsCPU').textContent = (analytics.cpu || 0).toFixed(1) + '%';
            $('#analyticsMemory').textContent = (analytics.memory || 0).toFixed(1) + '%';
            $('#analyticsVRAM').textContent = (analytics.vram || 0).toFixed(1) + '%';
            $('#analyticsDialogs').textContent = analytics.dialogs || 0;
            $('#analyticsGaps').textContent = analytics.gaps || 0;
            $('#analyticsLearned').textContent = analytics.learned || 0;
            
            // Graph metrics из /api/metrics
            const graphData = metrics.graph || {};
            const totalNodes = graphData.total_nodes || analytics.fractal_nodes || 0;
            const totalEdges = graphData.total_edges || analytics.fractal_edges || 0;
            const totalGroups = graphData.total_groups || analytics.fractal_groups || 0;
            
            $('#graphNodes').textContent = totalNodes;
            $('#graphEdges').textContent = totalEdges;
            $('#graphGroups').textContent = totalGroups;
            
            // Contradictions из graph.contradictions
            const contrad = graphData.contradictions || {};
            $('#contradTotal').textContent = contrad.total || graphData.contradictions || 0;
            $('#contradActive').textContent = contrad.active || 0;
            $('#contradResolved').textContent = (contrad.total || 0) - (contrad.active || 0);
            
            // Concepts from FGv2 nodes_by_type
            const nodesByType = graphData.nodes_by_type || {};
            $('#conceptExisting').textContent = nodesByType.concept || 0;
            $('#conceptProcessing').textContent = nodesByType.aci_concept || 0;
            $('#conceptCompleted').textContent = nodesByType.response || 0;
            
            // Curator metrics
            $('#curatorCycles').textContent = analytics.curator_cycles || 0;
            $('#curatorState').textContent = analytics.curator_state || 'idle';
            const nextRun = analytics.curator_next_run;
            if (nextRun && nextRun > 0) {
                const date = new Date(nextRun * 1000);
                $('#curatorNextRun').textContent = date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' });
            } else {
                $('#curatorNextRun').textContent = '—';
            }
            
            // System metrics из /api/metrics
            const systemMetrics = metrics.system || {};
            const gauges = metrics.gauges || {};
            
            // Web Search / Tavily metrics
            const counters = metrics.counters || {};
            const webSearches = counters.web_searches || analytics.web_searches || 0;
            $('#tavilyRequestsTotal').textContent = analytics.tavily_requests || 0;
            $('#tavilyResponsesTotal').textContent = analytics.tavily_responses || 0;
            $('#webSearchesTotal').textContent = webSearches;
            
            // Wiki metrics
            if ($('#wikiQueries')) $('#wikiQueries').textContent = analytics.wiki_queries || 0;
            if ($('#wikiArticles')) $('#wikiArticles').textContent = analytics.wiki_articles || totalNodes;
            if ($('#wikiCached')) $('#wikiCached').textContent = analytics.wiki_cached || 0;
            
            // System Health
            const healthStatus = $('#healthStatus');
            if (healthStatus) {
                const healthDot = healthStatus.querySelector('.health-dot');
                const healthText = healthStatus.querySelector('.health-text');
                const healthIssues = $('#healthIssues');
                
                // Определяем статус по метрикам
                let health = 'healthy';
                let issues = [];
                
                if ((analytics.cpu || 0) > 90) {
                    health = 'degraded';
                    issues.push('Высокая нагрузка CPU');
                }
                if ((analytics.memory || 0) > 90) {
                    health = 'degraded';
                    issues.push('Высокое потребление памяти');
                }
                if (totalNodes === 0) {
                    issues.push('Граф знаний пуст');
                }
                
                if (healthDot) healthDot.className = 'health-dot ' + health;
                if (healthText) healthText.textContent = health === 'healthy' ? 'Здоров' : 'Деградация';
                if (healthIssues) {
                    if (issues.length > 0) {
                        healthIssues.innerHTML = '<ul>' + issues.map(i => `<li>${esc(i)}</li>`).join('') + '</ul>';
                    } else {
                        healthIssues.innerHTML = '<span style="color: #4caf50">Проблем не обнаружено</span>';
                    }
                }
            }
            
            // Render activities
            const activityList = $('#activityList');
            if (analytics.activities && analytics.activities.length > 0) {
                activityList.innerHTML = analytics.activities.map(a => `
                    <div class="activity-item">
                        <div class="activity-icon">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                ${a.icon === 'memory' ? '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>' : 
                                  a.icon === 'learn' ? '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>' :
                                  '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>'}
                            </svg>
                        </div>
                        <div class="activity-content">
                            <div class="activity-title">${esc(a.title)}</div>
                            <div class="activity-time">${esc(a.time)}</div>
                        </div>
                    </div>
                `).join('');
            } else {
                activityList.innerHTML = '<div class="empty-state">Нет данных</div>';
            }
        }).catch(() => {
            toast('Ошибка загрузки аналитики', 'error');
        });
    }
    
    $('#refreshAnalytics')?.addEventListener('click', loadAnalytics);

    /* ── Learning ── */
    function loadLearning() {
        api('/learning').then(data => {
            if (data.error) {
                toast('Ошибка загрузки данных обучения', 'error');
                return;
            }
            
            // Update stats
            $('#learnTotal').textContent = data.total || 0;
            $('#learnSuccess').textContent = data.success || 0;
            $('#learnPending').textContent = data.pending || 0;
            
            // Render opportunities
            const oppList = $('#opportunitiesList');
            if (data.opportunities && data.opportunities.length > 0) {
                oppList.innerHTML = data.opportunities.map(o => `
                    <div class="opportunity-item">
                        <div class="opp-header">
                            <span class="opp-concept">${esc(o.concept)}</span>
                            <span class="opp-priority ${o.priority_level}">${(o.priority * 100).toFixed(0)}%</span>
                        </div>
                        <div class="opp-type">${esc(o.type)} • ${esc(o.domain)}</div>
                    </div>
                `).join('');
            } else {
                oppList.innerHTML = '<div class="empty-state">Нет доступных возможностей</div>';
            }
            
            // Render recent dialogs
            const dialogList = $('#dialogsList');
            if (data.recent_dialogs && data.recent_dialogs.length > 0) {
                dialogList.innerHTML = data.recent_dialogs.map(d => `
                    <div class="dialog-item">
                        <div class="dialog-header">
                            <span class="dialog-topic">${esc(d.topic)}</span>
                            <span class="dialog-outcome">${esc(d.outcome)}</span>
                        </div>
                    </div>
                `).join('');
            } else {
                dialogList.innerHTML = '<div class="empty-state">Нет недавних диалогов</div>';
            }
        }).catch(() => {
            toast('Ошибка загрузки обучения', 'error');
        });
    }
    
    $('#refreshLearning')?.addEventListener('click', loadLearning);

    /* ── System Info ── */
    function loadSystemInfo() {
        api('/system').then(data => {
            if (data.error) return;
            
            // Update model in settings
            const modelEl = document.querySelector('#settingsView .setting-group:last-child .setting-row:nth-child(2) span');
            if (modelEl) {
                modelEl.textContent = data.model;
            }
            
            // Update status indicator
            const statusDot = $('.status-dot');
            const statusText = $('.status-text');
            
            if (statusDot && statusText) {
                if (data.llama_cpp_ready || data.qwen_ready) {
                    statusDot.classList.add('active');
                    statusText.textContent = 'Готов';
                } else {
                    statusDot.classList.remove('active');
                    statusText.textContent = 'Загрузка...';
                }
            }
        }).catch(() => {});
    }
    
    // Check system status on load and periodically
    loadSystemInfo();
    setInterval(loadSystemInfo, 30000);
    
    /* ── Cache Stats ── */
    function loadCacheStats() {
        api('/cache-stats').then(data => {
            if (data.error) return;
            
            // Could display cache stats somewhere if needed
            console.log('Cache stats:', data);
        }).catch(() => {});
    }
    
    /* ── Settings toggles ── */
    async function loadSettings() {
        try {
            const d = await api('/settings');
            if (d.auto_learn !== undefined) settingsState.auto_learn = d.auto_learn;
            if (d.memory_enabled !== undefined) settingsState.memory_enabled = d.memory_enabled;
            if (d.sre_enabled !== undefined) settingsState.sre_enabled = d.sre_enabled;
            applySettingsToUI();
        } catch {
            const saved = localStorage.getItem('cogniflex_settings');
            if (saved) {
                try {
                    settingsState = JSON.parse(saved);
                    applySettingsToUI();
                } catch {}
            }
        }
    }

    function applySettingsToUI() {
        const al = $('#toggleAutoLearn');
        const me = $('#toggleMemory');
        const sre = $('#toggleSRE');
        if (al) al.classList.toggle('active', settingsState.auto_learn);
        if (me) me.classList.toggle('active', settingsState.memory_enabled);
        if (sre) sre.classList.toggle('active', settingsState.sre_enabled);
    }

    async function saveSettings() {
        try {
            await api('/settings', {
                method: 'POST',
                body: settingsState
            });
            localStorage.setItem('cogniflex_settings', JSON.stringify(settingsState));
        } catch {
            localStorage.setItem('cogniflex_settings', JSON.stringify(settingsState));
        }
    }

    function setupToggle(id, key) {
        const toggle = $(id);
        if (!toggle) return;
        toggle.addEventListener('click', () => {
            const isActive = !toggle.classList.contains('active');
            toggle.classList.toggle('active', isActive);
            settingsState[key] = isActive;
            saveSettings();
            toast(isActive ? 'Включено' : 'Выключено', 'info');
        });
    }
    
    setupToggle('#toggleAutoLearn', 'auto_learn');
    setupToggle('#toggleSRE', 'sre_enabled');
    setupToggle('#toggleMemory', 'memory_enabled');
    setupToggle('#toggleTheme', 'dark_theme');
    setupToggle('#toggleSound', 'sound_enabled');

    // Shutdown button
    $('#shutdownBtn')?.addEventListener('click', async () => {
        if (!confirm('Вы уверены, что хотите остановить EVA? Все сессии будут завершены.')) {
            return;
        }
        try {
            const result = await api('/api/shutdown', { method: 'POST' });
            toast('EVA останавливается...', 'info');
            // Close browser connection and notify user
            setTimeout(() => {
                alert('EVA система остановлена. Закройте эту вкладку браузера.');
                window.close();
            }, 1000);
        } catch (e) {
            toast('Ошибка остановки: ' + e.message, 'error');
        }
    });

    /* ── Export/Import ── */
    $('#exportSessionBtn')?.addEventListener('click', async () => {
        if (!activeSessionId) {
            toast('Нет активной сессии', 'error');
            return;
        }
        try {
            const d = await api('/export', { params: { session_id: activeSessionId, format: 'json' } });
            if (d.error) {
                toast('Ошибка экспорта: ' + d.error, 'error');
                return;
            }
            const blob = new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `session_${activeSessionId}_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
            toast('Сессия экспортирована', 'success');
        } catch {
            toast('Ошибка экспорта сессии', 'error');
        }
    });

    $('#importSessionBtn')?.addEventListener('click', () => {
        $('#importFileInput').click();
    });

    $('#importFileInput')?.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            const d = await api('/import', {
                method: 'POST',
                body: data
            });
            if (d.error) {
                toast('Ошибка импорта: ' + d.error, 'error');
                return;
            }
            if (d.sessions) sessions = d.sessions;
            if (d.session_id) activeSessionId = d.session_id;
            renderSessions();
            if (activeSessionId) await loadSessionMessages(activeSessionId);
            toast('Сессия импортирована', 'success');
        } catch {
            toast('Ошибка импорта. Проверьте формат файла.', 'error');
        }
        e.target.value = '';
    });

    /* ── Knowledge Panel ── */
    function loadKnowledge() {
        api('/knowledge').then(data => {
            if (data.error) return;
            
            $('#kgEntities').textContent = data.total_entities || data.entities?.length || 0;
            $('#kgRelations').textContent = data.total_relations || data.relations?.length || 0;
            
            if (activeSessionId && data.session_entities !== undefined) {
                $('#kgSessionEntities').textContent = data.session_entities;
            } else {
                $('#kgSessionEntities').textContent = '—';
            }
        }).catch(() => {});
    }

    function searchKnowledge(query) {
        if (!query.trim()) {
            $('#knowledgeResults').innerHTML = '<div class="empty-state">Введите запрос для поиска</div>';
            return;
        }
        
        api('/knowledge', {
            method: 'POST',
            body: { query: query.trim(), action: 'search' }
        }).then(data => {
            if (data.error) {
                $('#knowledgeResults').innerHTML = `<div class="empty-state">${esc(data.error)}</div>`;
                return;
            }
            
            const results = data.results || data.matches || [];
            if (results.length === 0) {
                $('#knowledgeResults').innerHTML = '<div class="empty-state">Ничего не найдено</div>';
                return;
            }
            
            $('#knowledgeResults').innerHTML = results.map(r => `
                <div class="kg-result-item">
                    <div class="kg-result-name">${esc(r.name || r.entity || r.keyword || 'Unknown')}</div>
                    <div class="kg-result-type">${esc(r.type || r.relation_type || '')}</div>
                    ${r.content || r.description ? `<div class="kg-result-content">${esc(r.content || r.description || '')}</div>` : ''}
                </div>
            `).join('');
        }).catch(() => {
            $('#knowledgeResults').innerHTML = '<div class="empty-state">Ошибка поиска</div>';
        });
    }

    $('#knowledgeSearchBtn')?.addEventListener('click', () => {
        searchKnowledge($('#knowledgeSearchInput').value);
    });

    $('#knowledgeSearchInput')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            searchKnowledge($('#knowledgeSearchInput').value);
        }
    });

    $('#refreshKnowledge')?.addEventListener('click', () => {
        loadKnowledge();
    });

    /* ── Enhanced Model Status ── */
    function loadModelStatus() {
        api('/model-status').then(data => {
            if (data.error) return;
            
            const models = data.models || {};
            const listEl = $('#modelStatusList');
            
            if (Object.keys(models).length > 0) {
                listEl.innerHTML = Object.entries(models).map(([key, model]) => `
                    <div class="model-item">
                        <div>
                            <div class="model-name">${esc(model.name || key)}</div>
                            <div class="model-role">${esc(model.role || '')}</div>
                        </div>
                        <span class="model-status-badge ${model.loaded ? 'loaded' : 'unloaded'}">
                            ${model.loaded ? 'Загружена' : 'Не загружена'}
                        </span>
                    </div>
                `).join('');
            } else {
                listEl.innerHTML = '<div class="empty-state">Нет данных о моделях</div>';
            }
            
            const gpuEl = $('#gpuInfo');
            if (data.gpu !== undefined || data.vram_usage !== undefined || data.gpu_available !== undefined) {
                const gpuAvail = data.gpu_available !== undefined ? data.gpu_available : (data.gpu !== undefined ? data.gpu : false);
                const vram = data.vram_usage || data.vram || 0;
                gpuEl.innerHTML = `
                    <div class="gpu-stat">
                        <span class="gpu-stat-val">${gpuAvail ? 'Да' : 'Нет'}</span>
                        <span class="gpu-stat-lbl">GPU</span>
                    </div>
                    <div class="gpu-stat">
                        <span class="gpu-stat-val">${typeof vram === 'number' ? vram.toFixed(1) + '%' : esc(String(vram))}</span>
                        <span class="gpu-stat-lbl">VRAM</span>
                    </div>
                `;
            }
            
            const pipelineEl = $('#pipelineStatus');
            if (pipelineEl) {
                pipelineEl.className = `pipeline-status ${data.pipeline_ready ? 'ready' : 'not-ready'}`;
                pipelineEl.textContent = data.pipeline_ready ? 'Пайплайн готов' : 'Пайплайн не готов';
            }
            
            const fm = data.fractal_memory || {};
            const fmEl = $('#fractalMemoryInfo');
            if (fmEl && fm.enabled) {
                fmEl.innerHTML = `
                    <span>Узлов: ${fm.nodes}</span>
                    <span>Связей: ${fm.edges}</span>
                    <span>Опытов: ${fm.experiences}</span>
                    <span>Концептов: ${fm.concepts}</span>
                `;
            }
        }).catch(() => {});
    }

    $('#modelStatusToggle')?.addEventListener('click', () => {
        const toggle = $('#modelStatusToggle');
        const body = $('#modelStatusBody');
        const isOpen = toggle.classList.toggle('open');
        body.classList.toggle('open', isOpen);
        if (isOpen) loadModelStatus();
    });
    
    loadModelStatus();
    setInterval(loadModelStatus, 60000);

    /* ── Self-Dialog ── */
    function loadSelfDialog() {
        api('/self-dialog').then(data => {
            if (data.error) return;
            
            const statusEl = $('#selfDialogStatus');
            if (statusEl) {
                statusEl.innerHTML = `
                    <div class="sd-stat">
                        <span class="sd-label">Всего диалогов</span>
                        <span class="sd-value">${data.total_dialogs || 0}</span>
                    </div>
                    <div class="sd-stat">
                        <span class="sd-label">Успешных</span>
                        <span class="sd-value">${data.successful || 0}</span>
                    </div>
                    <div class="sd-stat">
                        <span class="sd-label">Неудачных</span>
                        <span class="sd-value">${data.failed || 0}</span>
                    </div>
                `;
            }
        }).catch(() => {});
    }
    
    // Trigger self-dialog
    function triggerSelfDialog() {
        api('/self-dialog', {
            method: 'POST',
            body: {}
        }).then(data => {
            if (data.status === 'success') {
                toast('Самодиалог запущен: ' + (data.topic || 'общая тема'), 'success');
                loadSelfDialog();
            } else {
                toast('Ошибка запуска самодиалога: ' + (data.error || ''), 'error');
            }
        }).catch(() => {
            toast('Ошибка запуска самодиалога', 'error');
        });
    }
    
    $('#triggerSelfDialog')?.addEventListener('click', triggerSelfDialog);
    if ($('#selfDialogStatus')) loadSelfDialog();

    /* ── Snapshots ── */
    function loadSnapshots() {
        api('/snapshots').then(data => {
            if (data.error) return;
            
            const listEl = $('#snapshotList');
            if (listEl && data.snapshots) {
                if (data.snapshots.length > 0) {
                    listEl.innerHTML = data.snapshots.map(s => `
                        <div class="snapshot-item">
                            <span class="snapshot-name">${esc(s.name)}</span>
                            <span class="snapshot-info">${s.experiences} опытов, ${s.concepts} концептов</span>
                        </div>
                    `).join('');
                } else {
                    listEl.innerHTML = '<div class="empty-state">Нет слепков</div>';
                }
            }
        }).catch(() => {});
    }
    
    function createSnapshot() {
        api('/snapshots', {
            method: 'POST',
            body: {}
        }).then(data => {
            if (data.status === 'success') {
                toast('Слепок создан', 'success');
                loadSnapshots();
            } else {
                toast('Ошибка создания слепка', 'error');
            }
        }).catch(() => {
            toast('Ошибка создания слепка', 'error');
        });
    }
    
    $('#createSnapshot')?.addEventListener('click', createSnapshot);
    if ($('#snapshotList')) loadSnapshots();

    /* ─── Health Panel ─── */
    function loadHealth() {
        api('/status').then(data => {
            if (data.error) return;
            const cpu = data.cpu_usage || data.cpu || 0;
            const mem = data.memory_usage || data.memory || 0;
            $('#healthCPU').textContent = cpu.toFixed(1) + '%';
            $('#healthMemory').textContent = mem.toFixed(1) + '%';
            const cpuBar = $('#healthCPUBar');
            const memBar = $('#healthMemoryBar');
            cpuBar.style.width = Math.min(cpu, 100) + '%';
            memBar.style.width = Math.min(mem, 100) + '%';
            cpuBar.className = 'health-bar-fill' + (cpu > 80 ? ' danger' : cpu > 60 ? ' warn' : '');
            memBar.className = 'health-bar-fill' + (mem > 80 ? ' danger' : mem > 60 ? ' warn' : '');
        }).catch(() => {});

        api('/model-status').then(data => {
            if (data.error) return;
            const gpuAvail = data.gpu_available !== undefined ? data.gpu_available : (data.gpu || false);
            const vram = data.vram_usage || data.vram || 0;
            $('#healthGPU').textContent = gpuAvail ? 'Да' : 'Нет';
            $('#healthGPU').style.color = gpuAvail ? '#22c55e' : '#ef4444';
            $('#healthVRAM').textContent = typeof vram === 'number' ? vram.toFixed(1) + '%' : '—';
        }).catch(() => {});

        api('/system').then(data => {
            if (data.error) {
                $('#componentList').innerHTML = '<div class="empty-state">Не удалось загрузить</div>';
                return;
            }
            const components = [];
            if (data.brain_running !== undefined) components.push({ name: 'Brain', status: data.brain_running ? 'healthy' : 'unhealthy' });
            if (data.brain_connected !== undefined) components.push({ name: 'Brain Connection', status: data.brain_connected ? 'healthy' : 'unhealthy' });
            if (data.llama_cpp_ready !== undefined) components.push({ name: 'Llama.cpp', status: data.llama_cpp_ready ? 'healthy' : 'unhealthy' });
            if (data.qwen_ready !== undefined) components.push({ name: 'Qwen', status: data.qwen_ready ? 'healthy' : 'unhealthy' });
            if (data.model) components.push({ name: 'Model: ' + data.model, status: 'healthy' });

            if (components.length === 0) {
                Object.keys(data).forEach(key => {
                    const val = data[key];
                    if (typeof val === 'boolean') {
                        components.push({ name: key, status: val ? 'healthy' : 'unhealthy' });
                    }
                });
            }

            if (components.length === 0) {
                $('#componentList').innerHTML = '<div class="empty-state">Нет данных</div>';
                return;
            }

            const statusLabels = { healthy: 'OK', unhealthy: 'Ошибка', unknown: 'Неизвестно' };
            $('#componentList').innerHTML = components.map(c => `
                <div class="component-item">
                    <span class="component-name">${esc(c.name)}</span>
                    <span class="component-status ${c.status}">${statusLabels[c.status] || c.status}</span>
                </div>
            `).join('');
        }).catch(() => {
            $('#componentList').innerHTML = '<div class="empty-state">Ошибка загрузки</div>';
        });
    }

    $('#refreshHealth')?.addEventListener('click', loadHealth);

    /* ── Message Actions ── */
    function copyMessage(btn, text) {
        navigator.clipboard.writeText(text).then(() => {
            toast('Скопировано', 'success');
        }).catch(() => {
            toast('Ошибка копирования', 'error');
        });
    }
    
    function copyCodeBlock(btn) {
        const codeBlock = btn.closest('.code-block');
        const code = codeBlock.querySelector('code').textContent;
        navigator.clipboard.writeText(code).then(() => {
            btn.textContent = 'Скопировано!';
            setTimeout(() => { btn.textContent = 'Копировать'; }, 1500);
        }).catch(() => {
            toast('Ошибка копирования', 'error');
        });
    }
    
    function rateMessage(text, rating) {
        console.log('Rate message:', rating, text);
        toast(rating === 1 ? 'Спасибо за оценку!' : 'О учтём', 'success');
    }
    
    function regenerateMessage(msgDiv) {
        if (!msgDiv) return;

        const userMsgDiv = msgDiv.previousElementSibling;
        if (!userMsgDiv || !userMsgDiv.classList.contains('msg')) {
            toast('Не найден предыдущий запрос', 'error');
            return;
        }

        // Get the user message ID and text
        const userTextEl = userMsgDiv.querySelector('.msg-text');
        const userMsgId = userTextEl?.id?.replace('text-', '');
        const userText = userMsgId && messageStore[userMsgId] ? messageStore[userMsgId].text : userTextEl?.textContent;

        if (!userText) {
            toast('Не удалось получить текст запроса', 'error');
            return;
        }

        msgDiv.remove();
        // Remove the user message too since we're regenerating
        userMsgDiv.remove();
        sendMessage(userText);
    }

    /* ── Monitor ── */
    let monitorStream = null;
    let streamEnabled = true;
    let currentMonitorTab = 'all';
    
    function initMonitor() {
        const output = $('#monitorOutput');
        if (!output) return;
        
        // Setup tab filtering
        $$('.monitor-tab').forEach(tab => {
            tab.addEventListener('click', () => {
                $$('.monitor-tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                currentMonitorTab = tab.dataset.tab;
                filterMonitorOutput();
            });
        });
        
        // Clear button
        $('#clearMonitor')?.addEventListener('click', () => {
            output.innerHTML = '<div class="monitor-empty">Ожидание данных...</div>';
        });
        
        // Toggle stream
        $('#toggleStream')?.addEventListener('click', () => {
            streamEnabled = !streamEnabled;
            $('#toggleStream').textContent = streamEnabled ? 'Стрим: Вкл' : 'Стрим: Выкл';
            if (streamEnabled) {
                connectMonitorStream();
            } else {
                if (monitorStream) {
                    monitorStream.close();
                    monitorStream = null;
                }
            }
        });
        
        // Connect to event stream
        connectMonitorStream();
    }
    
    function connectMonitorStream() {
        if (monitorStream) {
            monitorStream.close();
        }
        
        if (!streamEnabled) return;
        
        const output = $('#monitorOutput');
        if (!output) return;
        
        const emptyEl = output.querySelector('.monitor-empty');
        if (emptyEl) {
            emptyEl.remove();
        }
        
        try {
            monitorStream = new EventSource('/api/events/stream');
            
            monitorStream.onmessage = (e) => {
                try {
                    const eventData = JSON.parse(e.data);
                    addMonitorLine(eventData);
                } catch (err) {
                    addMonitorLine({
                        event_type: 'raw',
                        data: { message: e.data }
                    });
                }
            };
            
            monitorStream.onerror = () => {
                setTimeout(() => {
                    if (streamEnabled) connectMonitorStream();
                }, 5000);
            };
        } catch (e) {
            console.error('SSE connection error:', e);
        }
    }
    
    function addMonitorLine(event) {
        const output = $('#monitorOutput');
        if (!output) return;
        
        const emptyEl = output.querySelector('.monitor-empty');
        if (emptyEl) {
            emptyEl.remove();
        }
        
        const eventType = event.event_type || 'unknown';
        const data = event.data || {};
        let type = 'generation';
        let message = '';
        
        // Format message based on event type and data
        if (eventType.includes('selfdialog') || eventType.includes('dialog')) {
            type = 'selfdialog';
            if (data.topic) {
                message = `Самодиалог по теме: "${data.topic}"`;
                if (data.outcome) message += ` → ${data.outcome}`;
                if (data.gaps?.length) message += ` (выявлено ${data.gaps.length} пробелов)`;
            } else if (data.role) {
                message = `Роль ${data.role}: ${data.message || data.content || 'активность'}`;
            } else {
                message = data.message || data.text || 'Самодиалог активен';
            }
        } else if (eventType.includes('learning') || eventType.includes('train')) {
            type = 'learning';
            if (data.source && data.content) {
                message = `Изучение из ${data.source}: "${data.content.substring(0, 50)}..."`;
            } else if (data.knowledge_gap) {
                message = `Выявлен пробел в знаниях: ${data.knowledge_gap}`;
            } else if (data.concept) {
                message = `Создан концепт: "${data.concept}"`;
            } else {
                message = data.message || data.status || 'Процесс обучения';
            }
        } else if (eventType.includes('generation') || eventType.includes('pipeline')) {
            type = 'generation';
            if (data.query) {
                message = `Генерация ответа для: "${data.query.substring(0, 40)}..."`;
            } else if (data.model) {
                message = `Модель ${data.model}: ${data.status || 'обработка'}`;
            } else {
                message = data.message || data.status || 'Генерация';
            }
        } else if (eventType.includes('error')) {
            type = 'error';
            message = data.error || data.message || 'Ошибка системы';
        } else if (eventType.includes('curator')) {
            type = 'curator';
            if (data.nodes_curated) {
                message = `Куратор обработал ${data.nodes_curated} узлов`;
            } else if (data.links_created) {
                message = `Создано ${data.links_created} новых связей`;
            } else {
                message = data.message || data.status || 'Куратор активен';
            }
        } else if (eventType.includes('contradiction')) {
            type = 'learning';
            if (data.contradiction_id) {
                message = `Обнаружено противоречие #${data.contradiction_id}`;
            } else {
                message = data.message || 'Анализ противоречий';
            }
        } else if (eventType.includes('concept')) {
            type = 'learning';
            if (data.concept_name) {
                message = `Концепт "${data.concept_name}" ${data.status || 'обрабатывается'}`;
            } else {
                message = data.message || 'Работа с концептами';
            }
        } else if (eventType.includes('web_search') || eventType.includes('tavily')) {
            type = 'generation';
            if (data.query) {
                message = `Веб-поиск: "${data.query.substring(0, 40)}..."`;
            } else if (data.results_count !== undefined) {
                message = `Найдено ${data.results_count} результатов`;
            } else {
                message = data.message || 'Поиск в интернете';
            }
        } else {
            // Generic formatting - try to extract meaningful info
            if (data.query) message = `Запрос: "${data.query.substring(0, 50)}..."`;
            else if (data.message) message = data.message;
            else if (data.content) message = data.content.substring(0, 100);
            else if (data.status) message = data.status;
            else message = JSON.stringify(data).substring(0, 100);
        }
        
        if (currentMonitorTab !== 'all' && type !== currentMonitorTab) {
            return;
        }
        
        const line = document.createElement('div');
        line.className = `monitor-line ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        const typeTag = `<span class="type-tag ${type}">${type}</span>`;
        
        line.innerHTML = `<span class="timestamp">[${timestamp}]</span>${typeTag}${esc(message)}`;
        
        output.appendChild(line);
        
        // Auto scroll
        output.scrollTop = output.scrollHeight;
        
        // Limit lines
        while (output.children.length > 500) {
            output.removeChild(output.firstChild);
        }
    }
    
    function filterMonitorOutput() {
        const output = $('#monitorOutput');
        if (!output) return;
        
        const lines = output.querySelectorAll('.monitor-line');
        lines.forEach(line => {
            if (currentMonitorTab === 'all') {
                line.style.display = '';
            } else {
                line.style.display = line.classList.contains(currentMonitorTab) ? '' : 'none';
            }
        });
    }
    
    // Also add polling fallback for learning events
    function startMonitorPolling() {
        setInterval(() => {
            if (!streamEnabled || !document.getElementById('monitorView')?.style.display.includes('flex')) return;
            
            fetch('/api/learning')
                .then(r => r.json())
                .then(data => {
                    if (data.opportunities?.length > 0) {
                        data.opportunities.forEach(op => {
                            addMonitorLine({
                                event_type: 'learning.opportunity',
                                data: { message: ` opportunity: ${op.concept}`, priority: op.priority }
                            });
                        });
                    }
                })
                .catch(() => {});
        }, 10000);
    }
    
    startMonitorPolling();

})();