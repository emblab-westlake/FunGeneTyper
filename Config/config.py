# -*-coding:utf-8-*-
Test_dict = {
    "ARGs":
        {"Category": 20,
         "label_list": {0: 'Betalactams', 1: 'Trimethoprim', 2: 'MLS', 3: 'Fusidic acid', 4: 'Fosfomycin',
                        5: 'Aminoglycosides', 6: 'Fluoroquinolones', 7: 'Multi-drug resistance', 8: 'Glycopeptides',
                        9: 'Phenicol', 10: 'Rifampin', 11: 'Tetracyclines', 12: 'Peptide', 13: 'Bacitracin',
                        14: 'Sulfonamide', 15: 'Nucleosides', 16: 'Aminocoumarins', 17: 'Triclosan', 18: 'Mupirocin',
                        19: 'Others'},
         "class_model": "Pretrained_model/ARGs_class.pkl",
         "group_model": "Pretrained_model/ARGs_group.pkl",
         "CoreEmbedding": 'Pretrained_model/resistance_gene/Core_DataBase/CoreEmbedding.txt',
         "Core": "Pretrained_model/resistance_gene/Core_DataBase/core.txt"
         },
        
    "VFs":
        {"Category": 46,
         "label_list": {0: 'Toxin', 1: 'Adherence and invasion', 2: 'Secretion system & effectors', 3: 'Motility',
                        4: 'Superantigen', 5: 'Iron uptake', 6: 'Regulation', 7: 'Immune evasion', 8: 'Capsule',
                        9: 'Antiphagocytosis', 10: 'Mammalian cell entry (mce) operons', 11: 'Endotoxin',
                        12: 'Serum resistance', 13: 'Autotransporter', 14: 'Unclassified', 15: 'Biofilm formation',
                        16: 'Antimicrobial activity', 17: 'Enzyme', 18: 'Immune modulator',
                        19: 'Variable surface lipoprotein', 20: 'Quorum sensing', 21: 'Cell surface components',
                        22: 'Magnesium uptake', 23: 'Glycosylation system', 24: 'Stress adaptation',
                        25: 'Catabolism of cholesterol', 26: 'Acid resistance', 27: 'Amino acid and purine metabolism',
                        28: 'Lipid and fatty acid metabolism', 29: 'Biosurfactant', 30: 'Manganese uptake',
                        31: 'Nutritional factor ', 32: 'Anti-apoptosis factor', 33: 'Anaerobic respiration',
                        34: 'Secreted proteins', 35: 'Copper uptake', 36: 'Phagosome arresting',
                        37: 'Intracellular survival', 38: 'Macrophage inducible genes', 39: 'Efflux pump',
                        40: 'Bile resistance', 41: 'Surface protein anchoring', 42: 'Actin-based motility',
                        43: 'Nucleation-promoting factor', 44: 'Peptidoglycan modification', 45: 'Non-VFs'},
         "class_model": "Pretrained_model/VFs_class.pkl",
         "group_model": "Pretrained_model/VFs_group.pkl",
         "CoreEmbedding": 'Pretrained_model/virulence_factor/Core_DataBase/CoreEmbedding.txt',
         "Core": "Pretrained_model/virulence_factor/Core_DataBase/core.txt"
         }
}
