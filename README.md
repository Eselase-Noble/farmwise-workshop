# FarmWise (Java)

This is the Java version of FarmWise, the smart farming assistant from the
Africa Digital Skills Conference workshop. It does the same two things the Python
project does, just written for the JVM:

1. **Crop recommendation.** A Random Forest model that suggests a crop from seven
   soil and weather numbers (N, P, K, temperature, humidity, pH, rainfall).
2. **Farming Q&A.** A retrieval based assistant that answers questions from
   farming guides using a local language model through Ollama, and cites the
   guide it used.

The Python project lives on the `ai` and `se` branches. This `java` branch is a
standalone port, so you will not find any Python here.

## What you need

* Java 21 or newer and Maven 3.9+
* [Ollama](https://ollama.com) running locally for the Q&A feature
* The crop dataset CSV in `data/raw/` (see `data/raw/README.md`)

## Getting started

```bash
# 1. Start Ollama and pull the two models (one time)
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Put the dataset at data/raw/Crop_recommendation.csv

# 3. Check that everything is in place
mvn -q exec:java -Dexec.mainClass=com.farmwise.app.CheckSetup

# 4. Train and save the crop model
mvn -q exec:java -Dexec.mainClass=com.farmwise.app.TrainModel

# 5. Run the full demo (crop prediction + a couple of RAG questions)
mvn -q exec:java -Dexec.mainClass=com.farmwise.app.Demo
```

## Using the code

Crop recommendation is a plain typed call:

```java
CropFeatures farm = new CropFeatures(90, 42, 43, 21, 82, 6.5, 200);
CropPrediction pred = Predictor.predictCrop(farm);
System.out.println(pred.crop() + " (" + pred.confidence() + ")");
System.out.println("out of range? " + pred.outOfDistribution());
```

Farming Q&A has two entry points that return the same answer type. One asks the
model directly, the other grounds the answer in the guides:

```java
Assistant assistant = new Assistant(store, new Embeddings(), new OllamaClient());
assistant.askWithoutRag("What is maize?");          // bare model, no sources
RagAnswer answer = assistant.ask("How much nitrogen does maize need?");
System.out.println(answer.answer());                // grounded, with citations
```

## How it is organised

```
src/main/java/com/farmwise/
  Config.java            paths, model names, chunk size and top_k from env vars
  llm/OllamaClient.java  small HTTP client for Ollama, with clear errors
  ml/
    CropData.java        load and validate the dataset
    Trainer.java         train, evaluate, save model and metadata
    Predictor.java       predictCrop(features) with top 3 and an out of range flag
    CropFeatures.java, CropPrediction.java
  rag/
    Loader.java          read .pdf/.txt/.md guides
    Chunker.java         simple overlapping chunks you can explain on a slide
    Embeddings.java      embeddings through Ollama, plus a cosine helper
    VectorStore.java     small interface: add, search, count, reset
    InMemoryVectorStore.java   the default backend (cosine search in memory)
    Indexer.java         load, chunk, embed, store
    Assistant.java       ask() and askWithoutRag()
  app/
    TrainModel.java, BuildIndex.java, CheckSetup.java, Demo.java
src/test/java/com/farmwise/   JUnit tests that run offline
data/                    guides and the dataset README (shared with the project)
```

## A note on the vector store

ChromaDB does not have an official Java client, so this port uses a small in
memory vector store that does cosine search. It implements the same
`VectorStore` interface (add, search, count, reset), which means a PostgreSQL
with pgvector backend can be dropped in later without touching the assistant.
That is the same design idea the Python project uses for Session 2.

## Tests

```bash
mvn test
```

The tests cover chunking, the vector store, the loader, prompt building and the
assistant, and they all run without Ollama or the dataset.

## Responsible AI

The crop dataset has no documented origin and is generally taken to reflect
Indian conditions, so it is not validated for Ghana. The predictor flags inputs
that fall outside the range it was trained on, and the Q&A assistant only answers
from the supplied guides and says it does not know when they do not cover the
question. The sample guides in `data/guides/` are for testing only and are not
real agronomic advice.

© Eselase-Noble. MIT License.
