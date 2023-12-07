import math
import os
import tkinter as tk
from pathlib import Path
from string import punctuation
from tkinter import filedialog, messagebox
import nltk
import yake
from langdetect import detect_langs
from nltk import sent_tokenize, word_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer

nltk.download('stopwords')

root = tk.Tk()
root.title("App")
root.configure(background='#d3d3d3')

doc_name = ""
text = ""
result = ""


def machine_learning(text, min_len, max_len):
    model_name = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_len, min_length=min_len, length_penalty=1.0,
                                 num_beams=8, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def name_of_document_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


def selecting_document_and_obtaining_contents():
    global doc_name, text
    resultText.delete('1.0', tk.END)
    file = filedialog.askopenfilename(multiple=False)
    doc_name = name_of_document_without_extension(file)
    text = Path(file, encoding="UTF-8", errors='ignore').read_text(encoding="UTF-8", errors='ignore')


def extract_keywords_from(text):
    languages = detect_langs(text)
    language = languages[0].lang if languages else 'en'
    kw_extractor = yake.KeywordExtractor(lan=language, n=3, dedupLim=0.3, top=15)
    keywords = kw_extractor.extract_keywords(text)
    return ', '.join([i[0] for i in keywords])


def get_essay(text):
    sentences = sent_tokenize(text)
    clean_sentences = []

    for sentence in sentences:
        terms = [term.lower() for term in word_tokenize(sentence) if term not in punctuation]
        clean_sentences.append(terms)

    scores = []
    for sentence in clean_sentences:
        score = 0
        for term in sentence:
            term_frequency = sentence.count(term) / len(sentence)
            max_frequency = max([s.count(term) for s in clean_sentences])
            inverse_document_frequency = math.log(len(clean_sentences) / sum([1 for s in clean_sentences if term in s]))
            score += term_frequency * (0.5 + (0.5 * term_frequency / max_frequency)) * inverse_document_frequency
        scores.append(score)

    summary_length = int(len(sentences) / 3)
    selected_sentences = []
    for _ in range(summary_length):
        max_score_index = scores.index(max(scores))
        selected_sentences.append(sentences[max_score_index])
        scores[max_score_index] = -1

    essay = '\n'.join(selected_sentences) if selected_sentences else "The text is too short."
    return essay


def document_result_information():
    global result
    result = f"Doc: {doc_name}\n"
    result += "KEY WORDS:\n"
    result += extract_keywords_from(text)
    result += "\n\nEssay:\n"
    result += get_essay(text)
    result += "\n\nMachine Learning:\n"
    result += machine_learning(text, int(len(text.split()) / 4), int(len(text.split()) / 3))
    resultText.configure(state='normal')
    resultText.insert('end', result)


def save_result_document():
    file = open(doc_name + '_result.txt', 'w', encoding="utf8")
    file.write(result)
    file.close()


def information_button():
    messagebox.showinfo("Lab 3",
                        "Choose a file to generate a summary, using Open doc, "
                        "then press the Get key words and summarize button.\n"
                        "To save the result, press the Save the result.")

aboutButton = tk.Button(root, text='Information about program', width=55, height=2, bg='#ff9999', font=('Arial', 12))
chooseDocButton = tk.Button(root, text='Open document', width=55, height=2, bg='#99ff99', font=('Arial', 12))
detectButton = tk.Button(root, text='Get key words and summarize', width=55, height=2, bg='#99ccff', font=('Arial', 12))
saveButton = tk.Button(root, text='Save the result', width=55, height=2, bg='#ffff99', font=('Arial', 12))
resultText = tk.Text(root, state='disabled', width=80, height=20, bg='#f0f0f0', font=('Arial', 11))

aboutButton.config(command=information_button)
chooseDocButton.config(command=selecting_document_and_obtaining_contents)
detectButton.config(command=document_result_information)
saveButton.config(command=save_result_document)

aboutButton.grid(row=0, column=0, pady=5)
chooseDocButton.grid(row=1, column=0, pady=5)
detectButton.grid(row=2, column=0, pady=5)
saveButton.grid(row=3, column=0, pady=5)
resultText.grid(row=4, column=0, pady=5)

root.mainloop()
