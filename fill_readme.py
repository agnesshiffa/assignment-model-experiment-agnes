"""Isi angka hasil Gemini ke README.md dari results/model_comparison_summary.csv.
Jalankan dari root project:  python3 fill_readme.py"""
import pandas as pd

df = pd.read_csv('results/model_comparison_summary.csv').set_index('Pendekatan')
few = pd.read_csv('results/llm_predictions_few_shot.csv')
n_err = int((few['raw_output'] == 'ERROR').sum())

def f(x):
    return f'{x:.4f}'

def row(name):
    r = df.loc[name]
    return (f"| {name} | {f(r['Accuracy'])} | {f(r['Precision_macro'])} | {f(r['Recall_macro'])} | "
            f"{f(r['F1_macro'])} | {f(r['Recall_negatif'])} | {int(r['Salah_prediksi'])} |")

def ops(name):
    r = df.loc[name]
    return (f"| {name} | Tidak | ±{r['Latensi per ulasan (ms)']:.0f} ms (batch 20 ulasan/request) | "
            f"~USD {r['Biaya per 1.000 ulasan (USD)']:.4f} |")

zs, fs = 'LLM Gemini (zero-shot)', 'LLM Gemini (few-shot)'
note = ''
if n_err:
    note = (f' — pada run final, {n_err} ulasan few-shot tetap gagal setelah 6 kali retry sehingga '
            f'tercatat sebagai prediksi salah (invalid); skor few-shot karenanya lebih rendah dari kemampuan sebenarnya')
s = open('README.md', encoding='utf-8').read()
s = (s.replace('{{ZS_ROW}}', row(zs)).replace('{{FS_ROW}}', row(fs))
      .replace('{{ZS_OPS}}', ops(zs)).replace('{{FS_OPS}}', ops(fs))
      .replace('{{LLM_ACC}}', f"zero-shot {df.loc[zs, 'Accuracy']:.2f} dan few-shot {df.loc[fs, 'Accuracy']:.2f}")
      .replace('{{FS_NOTE}}', note))
open('README.md', 'w', encoding='utf-8').write(s)
print('README berhasil diisi.', f'(few-shot ERROR: {n_err})')
