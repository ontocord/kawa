default_label2label = {'LOCATION': 'LOC'}

default_label2label_sample = {'SOC_ECO_CLASS': 'NORP',
                           'RACE': 'NORP',
                           'POLITICAL_PARTY': 'NORP',
                           'UNION': 'NORP',
                           'RELIGION': 'NORP',
                           'RELIGION_MEMBER': 'NORP',
                           'POLITICAL_PARTY_MEMBER': 'NORP',
                           'UNION_MEMBER': 'NORP',
                           'LANGUAGE': 'NORP',
                           'GPE': 'LOC',
                           'LOCATION': 'LOC',                           
                           'FAC': 'LOC',
                           'MEDICAL_SYMPTOM': 'DISEASE',
                           'CHEMICAL_SUBSTANCE': 'SUBSTANCE',
                           'TITLE': 'OTHER',
                           'PERSON_PRONOUN': 'OTHER',
                           }

default_upper_ontology = {
        'PERSON': ['PERSON'],
        'PUBLIC_FIGURE': ['PUBLIC_FIGURE', 'PERSON'],
        'TITLE': ['TITLE', 'PERSON'],
        'PERSON_PRONOUN': ['PERSON_PRONOUN', 'PERSON'],
        'LOC': ['LOC'],
        'GPE': ['GPE', 'LOC'],
        'FAC': ['FAC', 'LOC'],
        'ADDRESS': ['ADDRESS', 'LOC'],
        'ORG': ['ORG'],
        'NORP': ['NORP', 'ORG'],
        'SOC_ECO_CLASS': ['SOC_ECO_CLASS', 'NORP', 'ORG'],
        'RACE': ['RACE', 'NORP', 'ORG'],
        'POLITICAL_PARTY': ['POLITICAL_PARTY', 'NORP', 'ORG'],
        'UNION': ['UNION', 'NORP', 'ORG'],
        'RELIGION': ['RELIGION', 'NORP', 'ORG'],
        'RELIGION_MEMBER': ['RELIGION_MEMBER', 'NORP', 'ORG'],
        'POLITICAL_PARTY_MEMBER': ['POLITICAL_PARTY_MEMBER', 'NORP', 'ORG'],
        'UNION_MEMBER': ['UNION_MEMBER', 'NORP', 'ORG'],
        'LANGUAGE': ['LANGUAGE', 'NORP', 'ORG'],
        'AGE': ['AGE'],
        'DISEASE': ['DISEASE'],
        'MEDICAL_SYMPTOM': ['MEDICAL_SYMPTOM','DISEASE'],
        'PRODUCT': ['PRODUCT'],
        'USER': ['USER'],
        'URL': ['URL'],
        'ID': ['ID'],
        'LICENSE_PLATE': ['LICENSE_PLATE', 'ID'],
        'PHONE': ['PHONE', 'ID',],
        'IP_ADDRESS': ['IP_ADDRESS', 'ID',],
        'ANIMAL': ['ANIMAL'],
        'FOOD' : ['FOOD'],
        'PLANT': ['PLANT'],
        'GENDER': ['GENDER'],
        'JOB': ['JOB'],
        'EVENT': ['EVENT'],
        'BIO_CHEM_ENTITY': ['BIO_CHEM_ENTITY'],
        'MEDICAL_THERAPY': ['MEDICAL_THERAPY'],
        'SUBSTANCE': ['SUBSTANCE'],
        'CHEMICAL_SUBSTANCE': ['CHEMICAL_SUBSTANCE', 'SUBSTANCE'],
        'LAW': ['LAW'],
        'ANAT': ['ANAT'], #Anatomy
        'QUANTITY': ['QUANTITY'],
        'DATE': ['DATE'],
        'TIME': ['TIME'],
        'MISC': ['MISC'],
        'OTHER': ['OTHER'],
    }
