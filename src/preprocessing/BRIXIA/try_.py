import pydicom as dicom



image_PATH = '/Users/filruff/Desktop/PHD/PROGETTI/ITA-CINA/ANNO1/CODICE_ITA_CHINA/MultiObjective_BRIXIA-AIforCOVID/data/AIforCOVID/imgs/P_282.dcm'
dcm = dicom.dcmread(image_PATH)

array = dcm.pixel_array.astype(float)

print(array.shape)