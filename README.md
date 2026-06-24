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
[ Raw Dataset ] ➔ [ Feature Selection (6 Core Columns) ] ➔ [ NLTK NLP Pipeline ]
│
[ Streamlit UI ] 🔀 [ Cosine Similarity Engine ] 🔀 [ TF-IDF Vectorization ]
```

1. **Feature Reduction & Selection:** Narrowing down the raw dataset into 6 high-impact semantic and mathematical features for lean computational overhead.
2. **Text Preprocessing Pipeline (`nltk`):** Tokenization, stop-word elimination, case folding, and WordNet Lemmatization.
3. **Feature Engineering:** Constructing a comprehensive "Metadata Soup" combining text attributes (`overview`, `tagline`) and parsing structural JSON components (`genres`).
4. **Vectorization:** Transforming corpus text into numerical feature matrices using Term Frequency-Inverse Document Frequency (TF-IDF).
5. **Similarity Engine:** Measuring spatial distance using Cosine Similarity to serve real-time recommendations.

---

## 📊 Dataset Profile & Selected Features

The system operates on an optimized feature subset consisting of **45,453 rows** reduced down to **6 targeted columns**. This conscious reduction eliminates noise and reduces memory requirements during similarity matrix calculations.

### Optimized Schema Audit
| # | Column | Operational Dtype | Feature Category | Strategic NLP & Sorting Role |
|---|---|---|---|---|
| 0 | `title` | `object` (string) | Identifier | Target lookup key and UI display label. |
| 1 | `overview` | `object` (string) | Text Corpus | Primary text content for extracting core plot semantics and context. |
| 2 | `genres` | `object` (JSON/List) | Metadata | Parsed from stringified JSON lists to inject explicit thematic tags into the vocabulary soup. |
| 3 | `tagline` | `object` (string) | Text Corpus | Secondary contextual text feature to capture unique movie hooks and tones. |
| 4 | `vote_average`| `float64` | Quantitative | Statistical rating overlay used for quality thresholds or post-recommendation ranking. |
| 5 | `popularity`  | `float64` | Quantitative | Dynamic numerical metrics used for fallback trending sorts and popular filtering options. |

---

## 🧠 Deep NLP Engineering Pipeline

To make recommendations highly context-aware rather than just matching explicit keywords, text data passes through a stringent pipeline:

### 1. Advanced Text Preprocessing (`nltk`)
* **Tokenization & Regex Filtering:** Text is converted to lowercase, and structural symbols/punctuation are scrubbed out via alphanumeric regular expressions.
* **Stop-word Removal:** Custom-tuned `nltk.corpus.stopwords` (English) filters out structurally required but semantically empty filler words (*the, is, at, which*).
* **Lemmatization:** Leveraging `nltk.stem.WordNetLemmatizer` to reduce words to their morphological base form (e.g., *fights, fighting, fought* $\rightarrow$ *fight*). This prevents dimensional explosion in the sparse matrix.

### 2. Feature Synthesizer (The "Metadata Soup")
Instead of evaluating `overview` alone, the stringified JSON array in `genres` is programmatically extracted to clean strings (e.g., `[{'id': 16, 'name': 'Animation'}]` $\rightarrow$ `'Animation'`). This parsed text is then concatenated with the `overview` and `tagline` vectors to construct a single unified text profile block for every film.

### 3. Term Frequency-Inverse Document Frequency (TF-IDF)
The text block is vectorized into an $N \times M$ matrix where $N = 45453$ (movies) and $M = \text{vocabulary dimension}$.

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$

* We restrict parameters using `max_features` and set `ngram_range=(1, 2)` to capture both single terms and meaningful phrases (e.g., "science fiction", "romantic comedy").

### 4. Mathematical Similarity Matrix
Recommendations are retrieved by computing the **Cosine Similarity** score between the user's selected movie vector $\mathbf{A}$ and all other vectors $\mathbf{B}$ in the corpus.

$$\text{Cosine Similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

---

## ⚡ Production Optimizations (Memory & Speed)

Processing a $45453 \times 45453$ dense floating-point matrix can crash memory allocations ($>15 \text{ GB}$ RAM). This project implements enterprise-level optimizations:
* **Scipy Sparse Arrays:** Retaining vectors in `scipy.sparse` formats, dropping memory footprint by up to 85%.
* **Matrix Truncation:** Storing only the top 100 similarity indices for each movie during training, saving a compressed pickle file (`similarity_indices.pkl`) of less than $40\text{ MB}$ for deployment.
* **Streamlit Caching (`@st.cache_data`):** Prevents re-reading files or recalculating matrices on browser state updates.

---

## 🚀 Installation & Deployment

### Prerequisites
Make sure Python 3.9+ is installed on your environment.

### 1. Clone the repository
```bash
git clone (https://github.com/yourusername/cinematch-nlp.git)

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
 * **Hybridization:** Combining textual alignment with quantitative parameters like vote_average and popularity to perform a weighted score optimization.
 * **Transformer Migration:** Transitioning static TF-IDF weights to contextual dense vector embeddings via Sentence-BERT (all-MiniLM-L6-v2) or CodeBERT.
```

```
