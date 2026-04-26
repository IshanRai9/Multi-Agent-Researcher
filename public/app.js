// Agent node-name -> DOM element ID mapping
const AGENT_MAP = {
    'Searcher': 'agent-searcher',
    'Summarizer': 'agent-summarizer',
    'FactChecker': 'agent-factchecker',
    'Writer': 'agent-writer'
};

// Agent execution order for "next active" prediction
const AGENT_ORDER = ['Searcher', 'Summarizer', 'FactChecker', 'Writer'];

// Currently selected output length (default: standard)
let selectedLength = 'standard';

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

// --- Output Length Selector ---

document.querySelectorAll('.length-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.length-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedLength = btn.dataset.length;
    });
});

// --- DOCX Download ---

function downloadDocx() {
    const content = document.getElementById('reportContent');
    if (!content) return;

    const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType } = docx;

    const children = [];
    let inReferences = false;

    // Walk all child elements of reportContent
    for (const el of content.children) {
        const tag = el.tagName;
        const text = el.textContent.trim();
        if (!text) continue;

        // Detect references section
        if ((tag === 'H1' || tag === 'H2') && text.toLowerCase().includes('reference')) {
            inReferences = true;
        }

        if (tag === 'H1') {
            children.push(new Paragraph({
                children: [new TextRun({
                    text: text,
                    font: 'Times New Roman',
                    size: 44, // 22pt * 2 (half-points)
                    bold: true,
                })],
                heading: HeadingLevel.HEADING_1,
                spacing: { after: 200 },
            }));
        } else if (tag === 'H2') {
            children.push(new Paragraph({
                children: [new TextRun({
                    text: text,
                    font: 'Times New Roman',
                    size: 28, // 14pt * 2
                    bold: true,
                })],
                heading: HeadingLevel.HEADING_2,
                spacing: { before: 240, after: 120 },
            }));
        } else if (tag === 'H3') {
            children.push(new Paragraph({
                children: [new TextRun({
                    text: text,
                    font: 'Times New Roman',
                    size: 26, // 13pt
                    bold: true,
                })],
                heading: HeadingLevel.HEADING_3,
                spacing: { before: 200, after: 100 },
            }));
        } else if (tag === 'UL' || tag === 'OL') {
            const items = el.querySelectorAll('li');
            items.forEach(li => {
                const fontSize = inReferences ? 20 : 24; // 10pt or 12pt
                children.push(new Paragraph({
                    children: [new TextRun({
                        text: `• ${li.textContent.trim()}`,
                        font: 'Times New Roman',
                        size: fontSize,
                    })],
                    spacing: { after: 60 },
                    indent: { left: 720 }, // 0.5 inch indent
                }));
            });
        } else if (tag === 'P') {
            const fontSize = inReferences ? 20 : 24; // 10pt or 12pt
            children.push(new Paragraph({
                children: [new TextRun({
                    text: text,
                    font: 'Times New Roman',
                    size: fontSize,
                })],
                spacing: { after: 120 },
            }));
        } else {
            // Catch-all for other elements (blockquote, div, etc.)
            const fontSize = inReferences ? 20 : 24;
            children.push(new Paragraph({
                children: [new TextRun({
                    text: text,
                    font: 'Times New Roman',
                    size: fontSize,
                })],
                spacing: { after: 120 },
            }));
        }
    }

    const doc = new Document({
        sections: [{
            properties: {
                page: {
                    margin: {
                        top: 1440,    // 1 inch
                        right: 1440,
                        bottom: 1440,
                        left: 1440,
                    },
                },
            },
            children: children,
        }],
    });

    Packer.toBlob(doc).then(blob => {
        saveAs(blob, 'research_report.docx');
    });
}

// --- Research Flow ---

async function performResearch() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) return;

    // UI Updates
    const searchBtn = document.getElementById('searchBtn');
    const reportContent = document.getElementById('reportContent');
    const loader = document.getElementById('loader');
    const downloadBar = document.getElementById('downloadBar');

    searchBtn.disabled = true;
    reportContent.classList.add('hidden');
    downloadBar.classList.add('hidden');
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
                output_length: selectedLength
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
                            <h2>Connection Error</h2>
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
                        // Show download button
                        downloadBar.classList.remove('hidden');
                    }
                } catch (parseErr) {
                    console.warn('SSE parse error:', parseErr, raw);
                }
            }
        }
    } catch (err) {
        reportContent.innerHTML = `<div class="error-card">
            <h2>Server Error</h2>
            <p>Could not connect to the research server. Please ensure it is running.</p>
        </div>`;
    } finally {
        searchBtn.disabled = false;
        reportContent.classList.remove('hidden');
        loader.classList.add('hidden');
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
