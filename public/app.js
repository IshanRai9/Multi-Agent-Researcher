document.getElementById('searchBtn').addEventListener('click', async () => {
    const query = document.getElementById('queryInput').value;
    if (!query) return;

    // UI Updates
    document.getElementById('searchBtn').disabled = true;
    document.getElementById('reportContent').classList.add('hidden');
    document.getElementById('loader').classList.remove('hidden');
    
    // Animate agents to active
    const agents = document.querySelectorAll('#agentList li');
    agents.forEach(a => a.classList.add('active'));

    try {
        const response = await fetch('/research', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        const html = marked.parse(data.report || 'No content generated.');
        
        document.getElementById('reportContent').innerHTML = html;
    } catch (err) {
        document.getElementById('reportContent').innerHTML = `<p style="color:var(--error)">Error connecting to server.</p>`;
    } finally {
        document.getElementById('searchBtn').disabled = false;
        document.getElementById('reportContent').classList.remove('hidden');
        document.getElementById('loader').classList.add('hidden');
        agents.forEach(a => a.classList.remove('active'));
    }
});
