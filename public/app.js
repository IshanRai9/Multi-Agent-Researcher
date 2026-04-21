// Agent node-name → DOM element ID mapping
const AGENT_MAP = {
    'Searcher': 'agent-searcher',
    'Summarizer': 'agent-summarizer',
    'FactChecker': 'agent-factchecker',
    'Writer': 'agent-writer'
};

// Agent execution order for "next active" prediction
const AGENT_ORDER = ['Searcher', 'Summarizer', 'FactChecker', 'Writer'];

// Tracks the uploaded PDF's collection name (empty = no PDF)
let currentPdfCollection = '';

function resetAgents() {
    document.querySelectorAll('#agentList li').forEach(li => {
        li.classList.remove('active', 'done');
    });
}

function setAgentActive(name) {
    const id = AGENT_MAP[name];
    if (!id) return;
    const el = document.getElementById(id);
    if (el) {
        el.classList.remove('done');
        el.classList.add('active');
    }
}

function setAgentDone(name) {
    const id = AGENT_MAP[name];
    if (!id) return;
    const el = document.getElementById(id);
    if (el) {
        el.classList.remove('active');
        el.classList.add('done');
    }
}

function predictNextAgent(completedName) {
    const idx = AGENT_ORDER.indexOf(completedName);
    if (idx >= 0 && idx < AGENT_ORDER.length - 1) {
        return AGENT_ORDER[idx + 1];
    }
    return null;
}

// --- PDF Upload Handling ---

document.getElementById('pdfInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const statusEl = document.getElementById('uploadStatus');
    const labelEl = document.getElementById('uploadLabel');

    statusEl.classList.remove('hidden', 'upload-error', 'upload-success');
    statusEl.textContent = `Indexing "${file.name}"...`;
    statusEl.classList.add('upload-loading');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            statusEl.textContent = `❌ ${data.error}`;
            statusEl.classList.remove('upload-loading');
            statusEl.classList.add('upload-error');
            currentPdfCollection = '';
        } else {
            currentPdfCollection = data.collection;
            statusEl.textContent = `✅ "${data.filename}" indexed — will be used for RAG`;
            statusEl.classList.remove('upload-loading');
            statusEl.classList.add('upload-success');
            // Update label to show file is loaded
            labelEl.querySelector('.upload-text').textContent = `Replace PDF (current: ${data.filename})`;
        }
    } catch (err) {
        statusEl.textContent = '❌ Upload failed — server error';
        statusEl.classList.remove('upload-loading');
        statusEl.classList.add('upload-error');
        currentPdfCollection = '';
    }
});

// --- Research Flow ---

async function performResearch() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) return;

    // UI Updates
    const searchBtn = document.getElementById('searchBtn');
    const reportContent = document.getElementById('reportContent');
    const loader = document.getElementById('loader');

    searchBtn.disabled = true;
    reportContent.classList.add('hidden');
    loader.classList.remove('hidden');
    resetAgents();

    // Show first agent as active
    setAgentActive('Searcher');

    try {
        const response = await fetch('/research', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                pdf_collection: currentPdfCollection
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages from the buffer
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const raw = line.slice(6).trim();
                if (!raw) continue;

                try {
                    const data = JSON.parse(raw);

                    if (data.error) {
                        // Ollama connection error
                        reportContent.innerHTML = `<div class="error-card">
                            <h2>🔌 Connection Error</h2>
                            <p>${data.error}</p>
                        </div>`;
                        reportContent.classList.remove('hidden');
                        loader.classList.add('hidden');
                        searchBtn.disabled = false;
                        resetAgents();
                        return;
                    }

                    if (data.node) {
                        // Mark completed node as done
                        setAgentDone(data.node);
                        // Predict and activate the next agent
                        const next = predictNextAgent(data.node);
                        if (next) setAgentActive(next);
                    }

                    if (data.report) {
                        // Final report received
                        const html = marked.parse(data.report);
                        reportContent.innerHTML = html;
                    }
                } catch (parseErr) {
                    console.warn('SSE parse error:', parseErr, raw);
                }
            }
        }
    } catch (err) {
        reportContent.innerHTML = `<div class="error-card">
            <h2>🔌 Server Error</h2>
            <p>Could not connect to the research server. Please ensure it is running.</p>
        </div>`;
    } finally {
        searchBtn.disabled = false;
        reportContent.classList.remove('hidden');
        loader.classList.add('hidden');
        // Reset PDF collection after research (server cleans up the collection)
        currentPdfCollection = '';
        document.getElementById('uploadLabel').querySelector('.upload-text').textContent = 'Upload PDF for RAG (optional)';
        document.getElementById('uploadStatus').classList.add('hidden');
        document.getElementById('pdfInput').value = '';
    }
}

// Click handler
document.getElementById('searchBtn').addEventListener('click', performResearch);

// Enter key handler
document.getElementById('queryInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        performResearch();
    }
});
