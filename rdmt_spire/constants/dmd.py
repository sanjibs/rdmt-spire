from enum import StrEnum


class FileTypes(StrEnum):
    L2_SCIENCE = "science_wfi_level_2"
    L1_GUIDE_WINDOW = "guide_window_wfi_level_1"
    L4_DET_SOURCES = "science_wfi_level_4" #TODO: check what this will actually be
    L4_SOURCE_CATALOG_L2 = "science_wfi_level4_source_catalog_l2_product"

class ReprocessingStates(StrEnum):
    PROMPT = "none"
    REPROCESSED = "reprocessed"
    DATA_RELEASE = "data_release"

class ExposureTypes(StrEnum):
    WFI_IMAGE = "WFI_IMAGE"
    WFI_SPECTRAL = "WFI_SPECTRAL"
    WFI_IM_DARK = "WFI_IM_DARK"
    WFI_SP_DARK = "WFI_SP_DARK"
    WFI_FLAT = "WFI_FLAT"
    WFI_LOLO = "WFI_LOLO"
    WFI_WFSC = "WFI_WFSC"
    WFI_DARK = "WFI_DARK"
    WFI_GRISM = "WFI_GRISM"
    WFI_PRISM = "WFI_PRISM"
    UNEXPECTED = "UNEXPECTED"