import os
import pandas as pd

results_dir = 'results'

# Generate per-letter F1 table
if os.path.exists(os.path.join(results_dir, 'metrics_table.csv')):
    combined_df = pd.read_csv(os.path.join(results_dir, 'metrics_table.csv'), index_col=['Letter', 'dataset'])
    f1_table_data = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = {'Letter': letter}
        for ds in ['asl', 'bsl', 'isl']:
            if (letter, ds) in combined_df.index:
                f1 = combined_df.loc[(letter, ds), 'f1-score'] * 100
                row[f'{ds.upper()} F1 (%)'] = f"{f1:.2f}"
            else:
                row[f'{ds.upper()} F1 (%)'] = '–'
        f1_table_data.append(row)
    f1_table_df = pd.DataFrame(f1_table_data)
    print("\nPer-Letter F1 Scores")
    print(f1_table_df.to_string(index=False))
    f1_table_df.to_csv(os.path.join(results_dir, 'per_letter_f1.csv'), index=False)

# Generate selected language table
if os.path.exists(os.path.join(results_dir, 'selected_letters_summary.csv')):
    selected_df = pd.read_csv(os.path.join(results_dir, 'selected_letters_summary.csv'))
    selected_table_data = []
    for letter in sorted('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        row = selected_df[selected_df['Letter'] == letter]
        if not row.empty:
            lang = row['Selected Dataset'].values[0]
            selected_table_data.append({'Letter': letter, 'Selected Language': lang})
        else:
            selected_table_data.append({'Letter': letter, 'Selected Language': '–'})
    selected_table_df = pd.DataFrame(selected_table_data)
    print("\nSelected Language per Letter")
    print(selected_table_df.to_string(index=False))
    selected_table_df.to_csv(os.path.join(results_dir, 'selected_language_per_letter.csv'), index=False)

print("Tables generated.")