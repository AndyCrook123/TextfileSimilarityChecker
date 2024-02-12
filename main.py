import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

folder_path = 'testfiles'
sample_files = sorted([doc for doc in os.listdir(folder_path) if doc.endswith('.txt')])

sample_contents = []
for File in sample_files:
    file_path = os.path.join(folder_path, File)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            sample_contents.append(file.read())
    else:
        print(f"File not found: {file_path}")
        
vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

vectors = vectorize(sample_contents)
s_vectors = list(zip(sample_files, vectors))

def check_plagiarism():
    results = []
    global s_vectors
    for sample_n, text_vector_a in s_vectors:
        for sample_b, text_vector_b in s_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            sim_score = round(sim_score, 2)
            score = (sample_n, sample_b, sim_score)
            results.append(score)
    return results

for data in check_plagiarism():
    print(data)
    
import json

def create_graph_data(similarity_data, sample_files):
    nodes = [{"id": filename, "group": 1} for filename in sample_files]
    links = [{"source": source, "target": target, "value": score} for source, target, score in similarity_data]
    return {"nodes": nodes, "links": links}

def create_matrix_data(similarity_data, sample_files):
    file_index = {file: idx for idx, file in enumerate(sample_files)}
    matrix = [[0 for _ in sample_files] for _ in sample_files]

    for source, target, score in similarity_data:
        i, j = file_index[source], file_index[target]
        matrix[i][j] = score
        matrix[j][i] = score  # Since the matrix is symmetric

    # Set diagonal (self-similarity) to 1
    for i in range(len(sample_files)):
        matrix[i][i] = 1

    return {
        'matrix': matrix,
        'names': sample_files
    }

# Assuming check_plagiarism() function returns the similarity scores
similarity_data = check_plagiarism()

# Create data for graph visualization
graph_data = create_graph_data(similarity_data, sample_files)

# Create data for matrix visualization
matrix_data = create_matrix_data(similarity_data, sample_files)

# Write graph data to JSON file
with open('similarity_graph.json', 'w') as outfile:
    json.dump(graph_data, outfile, indent=4)

# Write matrix data to JSON file
with open('similarity_matrix.json', 'w') as outfile:
    json.dump(matrix_data, outfile, indent=4)
