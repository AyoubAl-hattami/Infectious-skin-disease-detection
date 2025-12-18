"""
Regenerate duplicate JSON files from existing quality assessment results.

Run: python regenerate_duplicate_jsons.py
"""

from data_quality_assessment import run_complete_assessment

diseases = ['Cellulitis', 'Folliculitis', 'Impetigo', 'Tinea']
base_path = 'Official data'

print("="*80)
print("REGENERATING DUPLICATE JSON FILES")
print("="*80)

for disease in diseases:
    dataset_path = f'{base_path}/{disease}'
    print(f"\n{'#'*80}")
    print(f"# Processing: {disease}")
    print(f"{'#'*80}")
    
    qa_results = run_complete_assessment(
        dataset_path=dataset_path,
        disease_name=disease,
        expected_source='Derm1M',
        dup_threshold=5,
        blur_threshold=100,
        output_dir=f'reports/{disease}'
    )

print("\n" + "="*80)
print("✅ ALL DUPLICATE JSON FILES GENERATED")
print("="*80)
print("\nGenerated files:")
for disease in diseases:
    print(f"  - reports/{disease}/{disease}_duplicates_dhash.json")
