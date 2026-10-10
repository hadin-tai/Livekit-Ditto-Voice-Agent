const API_URL = 'http://localhost:8000';

export async function getToken(identity, roomName, metadata = null) {
  const response = await fetch(`${API_URL}/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ identity, room_name: roomName, metadata }),
  });

  if (!response.ok) {
    throw new Error('Failed to get token');
  }

  return response.json();
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch(`${API_URL}/api/documents/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload document');
  }

  return response.json();
}

export async function listDocuments() {
  const response = await fetch(`${API_URL}/api/documents`);
  if (!response.ok) {
    throw new Error('Failed to list documents');
  }
  return response.json();
}

export async function deleteDocument(documentId) {
  const response = await fetch(`${API_URL}/api/documents/${documentId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error('Failed to delete document');
  }
  return response.json();
}
