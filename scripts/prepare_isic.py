from pathlib import Path


root = Path("/home/patsias/Master_Thesis/ISIC_dataset2018")
assert (root / "ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input").exists(), "Missing Training Input"
assert (root / "ISIC2018_Task1_Training_GroundTruth/ISIC2018_Task1_Training_GroundTruth").exists(), "Missing Training GT"
assert (root / "ISIC2018_Task1-2_Validation_Input/ISIC2018_Task1-2_Validation_Input").exists(), "Missing Validation Input"
assert (root / "ISIC2018_Task1_Validation_GroundTruth/ISIC2018_Task1_Validation_GroundTruth").exists(), "Missing Validation GT"
assert (root / "ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input").exists(), "Missing Test Input"
assert (root / "ISIC2018_Task1_Test_GroundTruth/ISIC2018_Task1_Test_GroundTruth").exists(), "Missing Test GT"
print("✅ ISIC structure OK")