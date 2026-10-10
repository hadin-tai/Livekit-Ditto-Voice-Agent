import React, { useState, useEffect } from 'react';
import { uploadDocument, listDocuments, deleteDocument } from '../api';
import './DocumentManager.css';

const DocumentManager = () => {
  const [documents, setDocuments] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState(null);
  const [selectedDocName, setSelectedDocName] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);

  // Load saved selection from localStorage
  useEffect(() => {
    const savedDocId = localStorage.getItem('selectedDocumentId');
    const savedDocName = localStorage.getItem('selectedDocumentName');
    if (savedDocId) {
      setSelectedDocId(savedDocId);
      setSelectedDocName(savedDocName);
    }
  }, []);

  // Load documents from backend
  const loadDocuments = async () => {
    setLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch (error) {
      console.error('Failed to load documents:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  // Handle file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    try {
      await uploadDocument(file);
      await loadDocuments();
    } catch (error) {
      console.error('Failed to upload document:', error);
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  // Handle document selection
  const handleSelectDocument = (doc) => {
    setSelectedDocId(doc.document_id);
    setSelectedDocName(doc.document_name);
    localStorage.setItem('selectedDocumentId', doc.document_id);
    localStorage.setItem('selectedDocumentName', doc.document_name);
  };

  // Handle document deletion
  const handleDeleteDocument = async (docId) => {
    if (!confirm('Are you sure you want to delete this document?')) return;
    
    try {
      await deleteDocument(docId);
      if (selectedDocId === docId) {
        setSelectedDocId(null);
        setSelectedDocName(null);
        localStorage.removeItem('selectedDocumentId');
        localStorage.removeItem('selectedDocumentName');
      }
      await loadDocuments();
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  return (
    <div className="document-manager">
      <div className="document-header">
        <div className="header-left">
          <div className="header-icon">📚</div>
          <div className="header-text">
            <h1>Document Management</h1>
            <p>Upload and manage your documents for the AI assistant</p>
          </div>
        </div>
        <div className="header-right">
          <a href="/" className="back-button">
            ← Back to Voice Room
          </a>
        </div>
      </div>

      <main className="document-main">
        {/* Upload Section */}
        <section className="upload-section glass">
          <h2>Upload New Document</h2>
          <label className="upload-label">
            <div className="upload-icon">📄</div>
            <div className="upload-text">
              {uploading ? 'Uploading...' : 'Click or drag PDF here'}
            </div>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              disabled={uploading}
            />
          </label>
        </section>

        {/* Documents List */}
        <section className="documents-section glass">
          <h2>Your Documents</h2>
          {loading ? (
            <div className="loading-state">
              <div className="loading-spinner"></div>
              <p>Loading documents...</p>
            </div>
          ) : documents.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📂</div>
              <p>No documents uploaded yet</p>
            </div>
          ) : (
            <div className="documents-list">
              {documents.map((doc) => (
                <div
                  key={doc.document_id}
                  className={`document-item ${
                    selectedDocId === doc.document_id ? 'selected' : ''
                  }`}
                >
                  <div className="document-info">
                    <div className="doc-icon">📄</div>
                    <div className="doc-details">
                      <div className="doc-name">{doc.document_name}</div>
                      <div className="doc-id">ID: {doc.document_id.substring(0, 8)}...</div>
                    </div>
                  </div>
                  <div className="document-actions">
                    <button
                      onClick={() => handleSelectDocument(doc)}
                      className={`select-button ${
                        selectedDocId === doc.document_id ? 'active' : ''
                      }`}
                    >
                      {selectedDocId === doc.document_id ? 'Selected' : 'Select'}
                    </button>
                    <button
                      onClick={() => handleDeleteDocument(doc.document_id)}
                      className="delete-button"
                    >
                      🗑️
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Current Selection */}
        {selectedDocName && (
          <section className="selection-section glass">
            <h2>Active Document</h2>
            <div className="active-doc">
              <div className="active-doc-icon">✅</div>
              <div className="active-doc-name">{selectedDocName}</div>
            </div>
            <p className="selection-note">
              This document will be used for AI queries in the voice room.
            </p>
          </section>
        )}
      </main>
    </div>
  );
};

export default DocumentManager;
