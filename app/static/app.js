// Dual Assistant Dashboard Controller

document.addEventListener('DOMContentLoaded', () => {
    // State management
    let sessionId = getOrCreateSessionId();
    let localOllamaModels = [];
    let isOllamaOnline = false;

    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBothToggle = document.getElementById('send-both-toggle');
    const modeDescText = document.getElementById('mode-desc-text');
    const clearSessionBtn = document.getElementById('clear-session-btn');
    const sessionIdDisplay = document.getElementById('session-id-display');
    
    const frontierChatLog = document.getElementById('frontier-chat-log');
    const frontierLatency = document.getElementById('frontier-latency');
    const frontierStatusText = document.getElementById('frontier-status-text');

    const ossChatLog = document.getElementById('oss-chat-log');
    const ossLatency = document.getElementById('oss-latency');
    const ossModelSelector = document.getElementById('ollama-model-selector');
    const ossStatusIndicator = document.getElementById('oss-status-indicator');
    const ossStatusText = document.getElementById('oss-status-text');

    const toast = document.getElementById('toast');

    // Initialize Dashboard
    init();

    function init() {
        sessionIdDisplay.textContent = sessionId;
        
        // Load local Ollama models on startup
        fetchOllamaModels();

        // Setup event listeners
        chatForm.addEventListener('submit', handleFormSubmit);
        clearSessionBtn.addEventListener('click', handleClearSession);
        
        sendBothToggle.addEventListener('change', (e) => {
            if (e.target.checked) {
                modeDescText.textContent = "Query Both Simultaneously";
                toastMessage("💡 Unified query mode activated. Both assistants will reply.", "info");
            } else {
                modeDescText.textContent = "Query Frontier Only";
                toastMessage("💡 Single query mode. Only the Frontier assistant will reply.", "info");
            }
        });

        // Auto-resize textarea as user types
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        // Submit form on Enter key (without Shift)
        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                chatForm.requestSubmit();
            }
        });
    }

    // Session ID management
    function getOrCreateSessionId() {
        let sid = localStorage.getItem('dual_assistant_session_id');
        if (!sid) {
            sid = 'session_' + Math.random().toString(36).substring(2, 11);
            localStorage.setItem('dual_assistant_session_id', sid);
        }
        return sid;
    }

    // Fetch dynamic models from Ollama daemon
    async function fetchOllamaModels() {
        ossStatusText.textContent = "Connecting to Ollama...";
        try {
            const response = await fetch('/api/models/ollama');
            if (response.ok) {
                const data = await response.json();
                localOllamaModels = data.models || [];
                
                if (localOllamaModels.length > 0) {
                    isOllamaOnline = true;
                    populateOllamaDropdown(localOllamaModels);
                    ossStatusIndicator.className = 'status-indicator online';
                    ossStatusText.textContent = `${localOllamaModels.length} models detected.`;
                } else {
                    isOllamaOnline = false;
                    setOllamaOfflineState("No models found. Run 'ollama pull phi3:mini'.");
                }
            } else {
                throw new Error("API returned an error");
            }
        } catch (error) {
            isOllamaOnline = false;
            setOllamaOfflineState("Ollama daemon is offline.");
        }
    }

    function populateOllamaDropdown(models) {
        ossModelSelector.innerHTML = '';
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            // Select standard recommended ones by default if they exist
            if (model.includes('phi3') || model.includes('qwen') || model.includes('llama')) {
                option.selected = true;
            }
            ossModelSelector.appendChild(option);
        });
    }

    function setOllamaOfflineState(msg) {
        ossStatusIndicator.className = 'status-indicator offline';
        ossStatusText.textContent = msg;
        ossModelSelector.innerHTML = '<option value="" disabled selected>Ollama Unavailable</option>';
        toastMessage("⚠️ Local Ollama is offline or has no models installed. OSS panel will be restricted.", "error");
    }

    // Clear session memory handler
    async function handleClearSession() {
        if (confirm("Are you sure you want to clear the short-term memory and history for this session?")) {
            try {
                const response = await fetch('/api/session/clear', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: sessionId })
                });

                if (response.ok) {
                    // Reset Session ID
                    localStorage.removeItem('dual_assistant_session_id');
                    sessionId = getOrCreateSessionId();
                    sessionIdDisplay.textContent = sessionId;

                    // Clear logs
                    clearChatLogs();
                    toastMessage("🧹 Memory and logs wiped successfully!", "success");
                } else {
                    throw new Error("Failed to clear session on backend");
                }
            } catch (err) {
                toastMessage("❌ Failed to reset session on server.", "error");
            }
        }
    }

    function clearChatLogs() {
        frontierChatLog.innerHTML = `
            <div class="system-bubble">
                <p>🤖 Frontier Assistant context cleared. New session started.</p>
            </div>
        `;
        ossChatLog.innerHTML = `
            <div class="system-bubble">
                <p>💻 Open Source Assistant context cleared. New session started.</p>
            </div>
        `;
        frontierLatency.textContent = '--';
        ossLatency.textContent = '--';
    }

    // Submit prompt handler
    async function handleFormSubmit(e) {
        e.preventDefault();
        const prompt = userInput.value.trim();
        if (!prompt) return;

        // Reset input box
        userInput.value = '';
        userInput.style.height = 'auto';

        // 1. Add user bubble to active panels
        appendMessageBubble(frontierChatLog, 'user', prompt);
        if (sendBothToggle.checked) {
            appendMessageBubble(ossChatLog, 'user', prompt);
        }

        // Scroll chat logs to bottom
        scrollToBottom(frontierChatLog);
        scrollToBottom(ossChatLog);

        // 2. Add typing indicators to active panels
        const frontierIndicator = appendTypingIndicator(frontierChatLog);
        let ossIndicator = null;
        if (sendBothToggle.checked) {
            ossIndicator = appendTypingIndicator(ossChatLog);
        }

        // Disable input while processing
        toggleInputControls(true);

        // 3. Initiate API Promises
        const promises = [];

        // Promise for Frontier Assistant (Gemini)
        promises.push(
            queryAssistant("frontier", prompt)
                .then(data => {
                    removeElement(frontierIndicator);
                    appendMessageBubble(frontierChatLog, 'assistant', data.response);
                    frontierLatency.textContent = `${data.latency}s`;
                    scrollToBottom(frontierChatLog);
                })
                .catch(err => {
                    removeElement(frontierIndicator);
                    appendMessageBubble(frontierChatLog, 'assistant', `❌ Error: ${err.message}`);
                    frontierLatency.textContent = 'Err';
                    scrollToBottom(frontierChatLog);
                })
        );

        // Promise for OSS Assistant (Ollama) - only query if toggle is on
        if (sendBothToggle.checked) {
            const selectedModel = ossModelSelector.value;
            promises.push(
                queryAssistant("oss", prompt, selectedModel)
                    .then(data => {
                        removeElement(ossIndicator);
                        appendMessageBubble(ossChatLog, 'assistant', data.response);
                        ossLatency.textContent = `${data.latency}s`;
                        scrollToBottom(ossChatLog);
                    })
                    .catch(err => {
                        removeElement(ossIndicator);
                        appendMessageBubble(ossChatLog, 'assistant', `❌ Error: ${err.message}`);
                        ossLatency.textContent = 'Err';
                        scrollToBottom(ossChatLog);
                    })
            );
        }

        // Wait for all active promises to resolve
        await Promise.all(promises);
        toggleInputControls(false);
        userInput.focus();
    }

    async function queryAssistant(modelType, message, selectedModel = null) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message: message,
                model_type: modelType,
                selected_model: selectedModel
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Server communication failed");
        }

        return await response.json();
    }

    // UI Render Utilities
    function appendMessageBubble(logContainer, role, text) {
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${role}`;
        
        // Render with linebreaks / basic formatting
        const textNode = document.createElement('p');
        textNode.innerText = text;
        bubble.appendChild(textNode);
        
        logContainer.appendChild(bubble);
    }

    function appendTypingIndicator(logContainer) {
        const indicator = document.createElement('div');
        indicator.className = 'message-bubble assistant';
        
        const dotsWrapper = document.createElement('div');
        dotsWrapper.className = 'typing-indicator';
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dot.className = 'typing-dot';
            dotsWrapper.appendChild(dot);
        }
        
        indicator.appendChild(dotsWrapper);
        logContainer.appendChild(indicator);
        scrollToBottom(logContainer);
        return indicator;
    }

    function removeElement(element) {
        if (element && element.parentNode) {
            element.parentNode.removeChild(element);
        }
    }

    function scrollToBottom(container) {
        container.scrollTop = container.scrollHeight;
    }

    function toggleInputControls(disabled) {
        userInput.disabled = disabled;
        document.getElementById('send-btn').disabled = disabled;
        clearSessionBtn.disabled = disabled;
        sendBothToggle.disabled = disabled;
    }

    // Show floating notifications
    function toastMessage(message, type = "info") {
        toast.className = 'toast';
        toast.innerHTML = '';
        
        const icon = document.createElement('span');
        icon.textContent = type === "success" ? "✅" : type === "error" ? "❌" : "ℹ️";
        
        const text = document.createElement('span');
        text.textContent = message;
        
        toast.appendChild(icon);
        toast.appendChild(text);
        toast.classList.remove('hidden');

        setTimeout(() => {
            toast.classList.add('hidden');
        }, 5000);
    }
});
