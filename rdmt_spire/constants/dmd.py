from enum import StrEnum


class FileTypes(StrEnum):
    L2_SCIENCE = "science_wfi_level_2"
    L1_GUIDE_WINDOW = "guide_window_level_1"
    L1_GUIDE_WINDOW_FACE = "guide_window_level_1_face"
    L4_SOURCE_CATALOG_L2 = "science_wfi_level_4_source_catalog_l2_product"

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
