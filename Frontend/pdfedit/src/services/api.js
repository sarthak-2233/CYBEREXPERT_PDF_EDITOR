const API_BASE_URL = 'http://localhost:5000/api';

class PDFEditorAPI {
    async healthCheck() {
        const response = await fetch(`${API_BASE_URL}/health`);
        return await response.json();
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData,
        });

        return await response.json();
    }

    async addAnnotation(sessionId, annotation) {
        const response = await fetch(`${API_BASE_URL}/annotate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                annotation: annotation,
            }),
        });

        return await response.json();
    }

    async processVideo(sessionId, apiKey = '') {
        const response = await fetch(`${API_BASE_URL}/process-video`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId,
                openai_api_key: apiKey,
            }),
        });

        return await response.json();
    }

    async getSessionStatus(sessionId) {
        const response = await fetch(`${API_BASE_URL}/session/${sessionId}`);
        return await response.json();
    }

    async startScreenRecording() {
        const response = await fetch(`${API_BASE_URL}/record-screen`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        return await response.json();
    }

    async stopScreenRecording(sessionId) {
        const response = await fetch(`${API_BASE_URL}/stop-recording/${sessionId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        return await response.json();
    }

    async exportPDF(sessionId) {
        const response = await fetch(`${API_BASE_URL}/export-pdf/${sessionId}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `exported-${sessionId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            return { success: true };
        } else {
            return await response.json();
        }
    }
}

export const pdfEditorAPI = new PDFEditorAPI();