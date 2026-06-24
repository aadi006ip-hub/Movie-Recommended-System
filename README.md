# 🎬 Movie Recommendation System

An end-to-end Content-Based Movie Recommendation Engine built using Natural Language Processing (NLP). The system extracts semantic features from movie metadata, descriptions, and taglines using advanced text preprocessing (`nltk`) and statistical vectorization (`TF-IDF`) to map over 45,000 films into a high-dimensional geometric space for fast, accurate similarity retrieval.

---

## 🖥️ Application Dashboard

<div align="center">
  <img src="assets/dashboard_screenshot.png" alt="CineMatch Web Application UI" width="90%" style="border-radius: 8px; border: 1px solid #ddd;"/>
  <br>
  <sup><i>Figure 1: Interactive Streamlit UI dashboard featuring semantic natural-language title search and dynamic filtering configurations.</i></sup>
</div>
> 💡 **Developer Note:** To update this image, create an `assets/` directory in your root folder, save your application screenshot as `dashboard_screenshot.png`, and it will automatically map right here!

---

## 📌 Project Architecture & Pipeline


```
[ Raw Dataset ] ➔ [ Data Cleaning & Structuring ] ➔ [ NLTK NLP Pipeline ]
│
[ Streamlit UI ] 🔀 [ Cosine Similarity Engine ] 🔀 [ TF-IDF Vectorization ]
```

1. **Data Ingestion & Cleaning:** Parsing complex textual formats (e.g., JSON stringified lists in genres) and handling missing values across 45,466 entries.
2. **Text Preprocessing Pipeline (`nltk`):** Tokenization, stop-word elimination, case folding, and WordNet Lemmatization.
3. **Feature Engineering:** Constructing a comprehensive "Metadata Soup" combining `overview`, `tagline`, `genres`, and `keywords`.
4. **Vectorization:** Transforming corpus text into numerical feature matrices using Term Frequency-Inverse Document Frequency (TF-IDF).
5. **Similarity Engine:** Measuring spatial distance using Cosine Similarity to serve real-time recommendations.

---

## 📊 Dataset Profile

The system processes a comprehensive dataset consisting of **45,466 movies** across **24 operational columns**. 

### Dynamic Schema Audit
| # | Column | Non-Null Count | Dtype | NLP/Strategic Role |
|---|---|---|---|---|
| 2 | `budget` | 45466 | float | Feature engineering / Metadata filtering |
| 3 | `genres` | 45466 | object | Core Component for Metadata Soup |
| 5 | `id` | 45466 | int | Unique identifier for relational mapping |
| 7 | `original_language` | 45455 | object | Language filtering constraints |
| 8 | `original_title` | 45466 | object | Search query matching |
| 9 | `overview` | 44512 | object | Primary text corpus for semantic extraction |
| 10| `popularity` | 45461 | object | Dynamic ranking and post-recommendation sorting |
| 14| `release_date` | 45379 | object | Time-decay sorting / Epoch weighting |
| 19| `tagline` | 20412 | object | Secondary contextual text feature |
| 20| `title` | 45460 | object | Target UI Lookup key |

---

## 🧠 Deep NLP Engineering Pipeline

To make recommendations highly context-aware rather than just matching explicit keywords, text data passes through a stringent pipeline:

### 1. Advanced Text Preprocessing (`nltk`)
* **Tokenization & Regex Filtering:** Text is converted to lowercase, and structural symbols/punctuation are scrubbed out via alphanumeric regular expressions.
* **Stop-word Removal:** Custom-tuned `nltk.corpus.stopwords` (English) filters out structurally required but semantically empty filler words (*the, is, at, which*).
* **Lemmatization:** Leveraging `nltk.stem.WordNetLemmatizer` to reduce words to their morphological base form (e.g., *fights, fighting, fought* $\rightarrow$ *fight*). This prevents dimensional explosion in the sparse matrix.

### 2. Feature Synthesizer (The "Metadata Soup")
Instead of evaluating `overview` alone, columns like `genres`, `tagline`, and `production_companies` are transformed, formatted into single strings, and concatenated into a unified text feature block per movie.

### 3. Term Frequency-Inverse Document Frequency (TF-IDF)
The text block is vectorized into an $N \times M$ matrix where $N = 45466$ (movies) and $M = \text{vocabulary dimension}$.

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$

* We restrict parameters using `max_features` and set `ngram_range=(1, 2)` to capture both single terms and meaningful phrases (e.g., "science fiction", "romantic comedy").

### 4. Mathematical Similarity Matrix
Recommendations are retrieved by computing the **Cosine Similarity** score between the user's selected movie vector $\mathbf{A}$ and all other vectors $\mathbf{B}$ in the corpus.

$$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

---

## ⚡ Production Optimizations (Memory & Speed)

Processing a $45466 \times 45466$ dense floating-point matrix can crash memory allocations ($>15 \text{ GB}$ RAM). This project implements enterprise-level optimizations:
* **Scipy Sparse Arrays:** Retaining vectors in `scipy.sparse` formats, dropping memory foot-print by up to 85%.
* **Matrix Truncation:** Storing only the top 100 similarity indices for each movie during training, saving a compressed pickle file (`similarity_indices.pkl`) of less than $40\text{ MB}$ for deployment.
* **Streamlit Caching (`@st.cache_data`):** Prevents re-reading files or recalculating matrices on browser state updates.

---

## 🚀 Installation & Deployment

### Prerequisites
Make sure Python 3.9+ is installed on your environment.

### 1. Clone the repository
```bash
git clone (https://github.com/aadi006ip-hub/Movie-Recommended-System.git)

```
### 2. Install dependencies
```bash
pip install -r requirements.txt

```
> *Dependencies include: streamlit, pandas, numpy, scikit-learn, nltk*
> 
### 3. Initialize NLTK Assets
Run the interactive Python terminal or insert into your initialization script:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

```
### 4. Run Application
```bash
streamlit run app.py

```
## 🔮 Future Scalability Matrix
 * **Hybridization:** Integrating Collaborative Filtering via Matrix Factorization (SVD) to overlay textual alignment with actual user rating tendencies.
 * **Transformer Migration:** Transitioning static TF-IDF weights to contextual dense vector embeddings via Sentence-BERT (all-MiniLM-L6-v2) or CodeBERT.
```

```
