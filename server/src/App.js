import React, { useState, useEffect } from "react";

function App() {
  const [manuscript, setManuscript] = useState("");
  const [authors, setAuthors] = useState("");
  const [institutions, setInstitutions] = useState("");
  const [label, setLabel] = useState("");
  const [mapping, setMapping] = useState(null);
  const [results, setResults] = useState([]);
  const [coiResults, setCoiResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedDomain, setSelectedDomain] = useState("");
  const [showDomainSelection, setShowDomainSelection] = useState(false);
  const [isDomainAnalysisClicked, setIsDomainAnalysisClicked] = useState(false);
  const [isFindingReviewers, setIsFindingReviewers] = useState(false);
  const domains = [
    "Social Sciences",
    "Physical Sciences",
    "Health Sciences",
    "Life Sciences"
  ];

  const handleSubmit = async () => {
    console.log("🚀 Starting classification...");
    setIsProcessing(true);
    try {
      console.log("📤 Sending request to backend...");
      const res = await fetch("http://localhost:5001/run_pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manuscript, authors, institutions }),
      });

      console.log("📥 Response received, status:", res.status);
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      console.log("📦 Classification response:", data);
      
      if (data.error) {
        throw new Error(`Backend error: ${data.error}`);
      }
      
      setLabel(data.label);
      setMapping(data.mapping);
      setShowDomainSelection(true);
      // Clear previous results when starting fresh
      setResults([]);
      setCoiResults(null);
      
      console.log("✅ State updated - label:", data.label, "mapping:", data.mapping);
    } catch (error) {
      console.error("❌ ERROR in handleSubmit:", error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDomainAnalysis = async () => {
    if (!selectedDomain) {
      alert("Please select a domain first.");
      return;
    }
    
    setIsDomainAnalysisClicked(true);
    setIsFindingReviewers(true);  // Set processing state (turns button green)
    
    try {
      console.log("🔍 Sending domain analysis request with domain:", selectedDomain);
      const res = await fetch("http://localhost:5001/run_pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ manuscript, authors, institutions, domain: selectedDomain }),
      });
  
      const data = await res.json();
      console.log("📦 Domain analysis response:", data);
      setResults(data.authors);
      setCoiResults(data.coi_results);
    } catch (error) {
      console.error("ERROR", error);
      alert(`Error: ${error.message}`);
    } finally {
      setIsFindingReviewers(false);  // Reset processing state (turns button blue)
    }
  };

  const handleClearAndRestart = () => {
    setManuscript("");
    setAuthors("");
    setInstitutions("");
    setLabel("");
    setMapping(null);
    setResults([]);
    setCoiResults(null);
    setIsProcessing(false);
    setSelectedDomain("");
    setShowDomainSelection(false);
    setIsDomainAnalysisClicked(false);
    setIsFindingReviewers(false);
  };

  useEffect(() => {
    console.log("📦 Final results received:", results);
  }, [results]);

  useEffect(() => {
    console.log("🏷️ Label updated:", label);
  }, [label]);

  useEffect(() => {
    console.log("🗺️ Mapping updated:", mapping);
  }, [mapping]);

  useEffect(() => {
    console.log("👁️ Show domain selection:", showDomainSelection);
  }, [showDomainSelection]);

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Peer Reviewer Matching Platform</h1>

      <label style={styles.label}>Manuscript Abstract:</label>
      <textarea
        style={styles.textarea}
        rows={6}
        value={manuscript}
        onChange={(e) => setManuscript(e.target.value)}
        placeholder="Paste the manuscript abstract here..."
      />

      <label style={styles.label}>Author Names (one per line):</label>
      <textarea
        style={styles.textarea}
        rows={5}
        value={authors}
        onChange={(e) => setAuthors(e.target.value)}
        placeholder="Enter author names, one per line..."
      />

      <label style={styles.label}>Author Institutions (one per line):</label>
      <textarea
        style={styles.textarea}
        rows={4}
        value={institutions}
        onChange={e => setInstitutions(e.target.value)}
        placeholder="Enter institution names, one per line..."
      />

      <button 
        style={{
          ...styles.button,
          backgroundColor: isProcessing ? "#28a745" : "#0077cc",
          display: "flex",
          alignItems: "center",
          justifyContent: "center"
        }} 
        onClick={handleSubmit}
        disabled={isProcessing}
      >
        {isProcessing ? (
          <>
            <span className="spinner" style={{ marginRight: "10px" }}></span>
            Processing...
          </>
        ) : (
          "Run Classification"
        )}
      </button>

      {mapping && (
        <div style={styles.mappingContainer}>
          <h2 style={{ ...styles.title, fontSize: "20px", color: "#555", marginBottom: "15px" }}>
            Manuscript Classification Results
          </h2>
          <div style={styles.mappingGrid}>
            <div style={styles.mappingItem}>
              <strong>Topic ID:</strong> {mapping.topic_id}
            </div>
            <div style={styles.mappingItem}>
              <strong>Topic Name:</strong> {mapping.matched_topic_name}
            </div>
            <div style={styles.mappingItem}>
              <strong>Subfield:</strong> {mapping.subfield_name}
            </div>
            <div style={styles.mappingItem}>
              <strong>Field:</strong> {mapping.field_name}
            </div>
            <div style={styles.mappingItem}>
              <strong>Domain:</strong> {mapping.domain_name}
            </div>
            {mapping.keywords && mapping.keywords.length > 0 && (
              <div style={styles.mappingItem}>
                <strong>Keywords:</strong> {mapping.keywords.join(", ")}
              </div>
            )}
            {mapping.summary && (
              <div style={styles.mappingItem}>
                <strong>Summary:</strong> {mapping.summary}
              </div>
            )}
          </div>
        </div>
      )}

      {showDomainSelection && (
        <div style={styles.mappingContainer}>
          <h2 style={{ ...styles.title, fontSize: "20px", color: "#555", marginBottom: "15px" }}>
            Step 2: Find Reviewers
          </h2>
          <div style={{ 
            backgroundColor: "#e3f2fd", 
            padding: "15px", 
            borderRadius: "6px", 
            marginBottom: "20px",
            border: "1px solid #2196f3"
          }}>
            <strong>Next Step:</strong> Select a domain below and click "Find Reviewers" to get your personalized reviewer recommendations.
          </div>
          <h3 style={{ fontSize: "18px", marginBottom: "15px", color: "#555" }}>
            Restrict reviewers to one domain:
          </h3>
          <div style={{ marginBottom: "20px" }}>
            {domains.map((domain) => (
              <label key={domain} style={{ display: "block", marginBottom: "10px" }}>
                <input
                  type="radio"
                  value={domain}
                  checked={selectedDomain === domain}
                  onChange={() => setSelectedDomain(domain)}
                  style={{ marginRight: "10px" }}
                />
                {domain}
              </label>
            ))}
          </div>
          <button 
            style={{
              ...styles.button,
              backgroundColor: isFindingReviewers ? "#28a745" : "#0077cc",
              fontSize: "18px",
              padding: "12px 24px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }} 
            onClick={handleDomainAnalysis}
            disabled={!selectedDomain || isFindingReviewers}
          >
            {isFindingReviewers ? (
              <>
                <span className="spinner" style={{ marginRight: "10px" }}></span>
                Finding Reviewers...
              </>
            ) : (
              "Find Reviewers"
            )}
          </button>
        </div>
      )}

      {results.length > 0 && (
        <>
          {coiResults && (
            <div style={styles.mappingContainer}>
              <h2 style={{ ...styles.title, fontSize: "20px", color: "#555", marginBottom: "15px" }}>
                COI Test Results
              </h2>
              <div style={styles.coiSummary}>
                <div style={styles.coiItem}>
                  <strong>Total Authors Checked:</strong> {coiResults.total_checked}
                </div>
                <div style={styles.coiItem}>
                  <strong>Total Authors Rejected:</strong> {coiResults.total_rejected}
                </div>
                <div style={styles.coiItem}>
                  <strong>Direct Conflicts:</strong> {coiResults.direct_conflicts.length}
                </div>
                <div style={styles.coiItem}>
                  <strong>Coworker Conflicts:</strong> {coiResults.coworker_conflicts.length}
                </div>
              </div>
              
              {coiResults.rejected_authors.length > 0 && (
                <div style={styles.rejectedSection}>
                  <h3 style={{ fontSize: "16px", marginBottom: "10px", color: "#d32f2f" }}>
                    Rejected Authors
                  </h3>
                  <table style={styles.table}>
                    <thead>
                      <tr>
                        <th>Author</th>
                        <th>Conflict Type</th>
                        <th>Paper ID</th>
                      </tr>
                    </thead>
                    <tbody>
                      {coiResults.rejected_authors.map((author, i) => (
                        <tr key={i}>
                          <td>{author.author}</td>
                          <td style={styles.conflictType[author.conflict_type] || {}}>
                            {author.conflict_type.replace('_', ' ').toUpperCase()}
                          </td>
                          <td>{author.paper_id}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
          <div style={{ ...styles.mappingContainer, overflowX: 'auto' }}>
            <h2 style={{ ...styles.title, fontSize: "20px", color: "#555", marginBottom: "15px" }}>
              Top Ranked Reviewers
            </h2>
            <table style={{ ...styles.table, minWidth: '1200px' }}>
              <thead>
                <tr>
                  <th style={styles.tableCell}>Author</th>
                  <th style={styles.tableCell}>Score</th>
                  <th style={styles.tableCell}>Domain</th>
                  <th style={styles.tableCell}>Author ID</th>
                  <th style={styles.tableCell}>Paper ID</th>
                  <th style={styles.tableCell}>Similarity</th>
                  <th style={styles.tableCell}>Citations</th>
                  <th style={styles.tableCell}>Publications</th>
                  <th style={styles.tableCell}>Keywords</th>
                  <th style={styles.tableCell}>Abstract</th>
                  <th style={styles.tableCell}>OrcID</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td style={styles.tableCell}>{r.author}</td>
                    <td style={styles.tableCell}>{r.score?.toFixed(3)}</td>
                    <td style={styles.tableCell}>{r.domain || "Unknown"}</td>
                    <td style={styles.tableCell}>
                      <a 
                        href={r.author_id} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={styles.link}
                      >
                        {r.author_id}
                      </a>
                    </td>
                    <td style={styles.tableCell}>
                      <a 
                        href={r.paper_id}
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={styles.link}
                      >
                        {r.paper_id}
                      </a>
                    </td>
                    <td style={styles.tableCell}>{r.similarity?.toFixed(3)}</td>
                    <td style={styles.tableCell}>{r.cited?.toFixed(2)}</td>
                    <td style={styles.tableCell}>{r.works?.toFixed(2)}</td>
                    <td style={styles.tableCell}>{r.kw_jaccard?.toFixed(3)}</td>
                    <td style={styles.tableCell}>{r.abstract}</td>
                    <td style={styles.tableCell}>
                      <a 
                        href={r.orcid}
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={styles.link}
                      >
                        {r.orcid}
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* Clear and Restart button */}
      {(mapping || results.length > 0) && (
        <div style={{ textAlign: "center", marginTop: "30px" }}>
          <button 
            style={{
              ...styles.button,
              backgroundColor: "#dc3545",
              fontSize: "18px",
              padding: "15px 30px"
            }} 
            onClick={handleClearAndRestart}
          >
            Clear and Restart
          </button>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "800px",
    margin: "0 auto",
    fontFamily: "Segoe UI, sans-serif",
    padding: "40px 20px",
    color: "#333",
    backgroundColor: "#f9f9fb",
  },
  title: {
    textAlign: "center",
    fontSize: "32px",
    marginBottom: "30px",
    color: "#2a2a2a",
  },
  label: {
    fontWeight: "bold",
    marginTop: "20px",
    display: "block",
  },
  textarea: {
    width: "100%",
    padding: "10px",
    fontSize: "15px",
    borderRadius: "6px",
    border: "1px solid #ccc",
    marginBottom: "20px",
    resize: "vertical",
  },
  button: {
    padding: "10px 20px",
    fontSize: "16px",
    backgroundColor: "#0077cc",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    marginBottom: "40px",
  },
  resultsTitle: {
    fontSize: "24px",
    marginBottom: "15px",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    backgroundColor: "#fff",
    border: "1px solid #ddd",
    minWidth: '1200px',  // Minimum width to fit all columns
  },
  tableHeader: {
    backgroundColor: "#f1f1f1",
    fontWeight: "bold",
  },
  tableCell: {
    border: "1px solid #ddd",
    padding: "8px",
    maxWidth: "200px",    // Prevent cells from growing too wide
    wordWrap: "break-word", // Break long words
    whiteSpace: "normal",   // Allow wrapping
  },
  mappingContainer: {
    backgroundColor: "#f0f0f0",
    padding: "20px",
    borderRadius: "8px",
    marginBottom: "20px",
    overflowX: 'auto',  // Allows horizontal scrolling
    maxWidth: '100%',   // Ensures it doesn't overflow parent container
  },
  mappingGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
    gap: "10px",
  },
  mappingItem: {
    backgroundColor: "#fff",
    padding: "10px",
    borderRadius: "6px",
    border: "1px solid #eee",
  },
  link: {
    color: "#007bff",
    textDecoration: "none",
    cursor: "pointer",
  },
  coiSummary: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
    gap: "10px",
    marginBottom: "20px",
  },
  coiItem: {
    backgroundColor: "#fff",
    padding: "10px",
    borderRadius: "6px",
    border: "1px solid #eee",
    fontWeight: "bold",
  },
  rejectedSection: {
    marginTop: "20px",
    padding: "15px",
    backgroundColor: "#ffebee",
    borderRadius: "8px",
    border: "1px solid #ef9a9a",
  },
  conflictType: {
    "direct": { color: "#d32f2f", fontWeight: "bold" },
    "coworker": { color: "#388e3c", fontWeight: "bold" },
    "submitted_author": { color: "#1976d2", fontWeight: "bold" },
  },
};

export default App;