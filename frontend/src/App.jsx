import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];

    setSelectedFile(file);
    setResult(null);

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);

    setLoading(true);

    try {
      const response = await axios.post(API_URL, formData);
      setResult(response.data);
      await loadHistory();
    } catch (error) {
      console.error(error);
      alert("Prediction failed. Make sure FastAPI backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
  try {
    const response = await axios.get(
      "http://127.0.0.1:8000/history"
    );

    setHistory(response?.data);
  } catch (error) {
    console.error(error);
  }
};

useEffect(() => {
  loadHistory();
}, []);

  return (
    <main className="page">
      <section className="card">
        <p className="eyebrow">AI Computer Vision</p>
        <h1>Visual Defect Detector</h1>
        <p className="subtitle">
          Upload a product image. The model will classify it as normal or defective.
        </p>

        <label className="dropzone">
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <span>Click to upload an image</span>
          <small>PNG, JPG, or JPEG</small>
        </label>

        {previewUrl && (
          <img className="preview" src={previewUrl} alt="Uploaded preview" />
        )}

        <button onClick={handlePredict} disabled={!selectedFile || loading}>
          {loading ? (
            <span className="button-loading">
              <span className="spinner"></span>
              Analyzing...
            </span>
          ) : (
            "Analyze Image"
          )}
        </button>
        
        {result && <div className={`result ${result?.prediction}`}>
          <h2>{result?.prediction.toUpperCase()}</h2>

          <p>
            Defect Confidence:
            {" "}
            {(result?.confidence * 100).toFixed(2)}%
          </p>

          <p>
            Category:
            {" "}
            <strong>{result?.category}</strong>
          </p>

          <p>
            Category Confidence:
            {" "}
            {(result?.category_confidence * 100).toFixed(2)}%
          </p>
        </div>
        }

        {history.length > 0 && (
          <div className="history">
            <h3>Prediction History</h3>
            <div className="history-items">
              {history.map((item) => (
                <div className="history-item" key={item.id}>
                  <div>
                    <strong>{item.filename}</strong>
                    <p>{item.created_at}</p>
                  </div>

                  <div>
                    <span>{item.category}</span>
                    <span className={item.prediction}>{item.prediction}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}

export default App;