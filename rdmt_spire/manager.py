import json
import logging
import os
from importlib import import_module
from typing import Any, Dict, List

import asdf

from .constants.codes import StatusCodes
from .monitors.monitor_base import BaseMonitor

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MonitorManager:
    """
    MonitorManager is the main class that orchestrates the monitors and manages their
    input and output. It does not raise exception, in order to avoid abrupt termination
    instead it relies on error status codes for capturing errors. The monitors themselves
    can raise exceptions.

    """

    def __init__(self, input_file: asdf.AsdfFile | str, monitor_name: str, monitor_config: Dict[str, Any] = None):
        
        self.statusCode = StatusCodes.SUCCESS
        self.errors: list[str] = []
        self.log: list[str] = []
        self.asdf_file: asdf.AsdfFile | None = None
        self.monitor_objects: list[Any] = []

        if not isinstance(input_file, str):
            self.log.append(f"MonitorManager: initialized with input file: {input_file.uri}")
        else:
            self.log.append(f"MonitorManager: initialized with input file: {input_file}")

        try:
            # Extracting the asdf file tree
            if isinstance(input_file, asdf.AsdfFile):
                self.asdf_file = input_file

            elif os.path.exists(input_file):
                self.asdf_file = asdf.open(input_file)

            else:
                self.statusCode = StatusCodes.FAILURE
                self.errors.append(f"MonitorManager: Invalid ASDF input {input_file}")
                return

        except asdf.AsdfFileError as e:
            self.statusCode = StatusCodes.FAILURE
            self.errors.append(f"MonitorManager: failed to open ASDF file: {e}")

        if monitor_name == "essential":
            monitor_noise_1f = import_module("rdmt_spire.monitors.noise_1f")
            monitor_pixel = import_module("rdmt_spire.monitors.pixel")
            self.monitor_objects.append(
                monitor_noise_1f.Noise1fMonitor(self.asdf_file)
            )
            self.monitor_objects.append(
                monitor_pixel.PixelMonitor(self.asdf_file)
            )
        elif monitor_name == "noise_1f":
            monitor_noise_1f = import_module("rdmt_spire.monitors.noise_1f")
            self.monitor_objects.append(
                monitor_noise_1f.Noise1fMonitor(self.asdf_file)
            )

        elif monitor_name == "astrometry":
            if monitor_config is None or "astrometry" not in monitor_config or "datadir" not in monitor_config["astrometry"]:
                logger.error("MonitorManager: Missing 'datadir' in monitor_config for astrometry monitor.")
                self.statusCode = StatusCodes.FAILURE
                self.errors.append("MonitorManager: Missing 'datadir' in monitor_config for astrometry monitor.")
                return
            
            monitor_astrometry = import_module("rdmt_spire.monitors.astrometry")
            self.monitor_objects.append(
                monitor_astrometry.AstrometryMonitor(self.asdf_file, monitor_config["astrometry"]["datadir"])
            )
            
        elif monitor_name == "guide_window":
            monitor_guide_window = import_module("rdmt_spire.monitors.guide_window")
            self.monitor_objects.append(
                monitor_guide_window.GuideWindowMonitor(self.asdf_file)
            )

        elif monitor_name == "pixel":
            monitor_pixel = import_module("rdmt_spire.monitors.pixel")
            self.monitor_objects.append(
                monitor_pixel.PixelMonitor(self.asdf_file)
            )

        elif monitor_name == "source_catalog":
            config_dirs=["RDMT_SPIRE_LDATA_DIR", "RDMT_SPIRE_RDATA_DIR", "RDMT_SPIRE_L4_DIR"]
            if monitor_config is None or "source_catalog" not in monitor_config or len(set(config_dirs) - set(monitor_config["source_catalog"])) != 0:
                logger.error("MonitorManager: Missing required config_dirs in monitor_config for source_catalog monitor.")
                self.statusCode = StatusCodes.FAILURE
                self.errors.append("MonitorManager: Missing required config_dirs in monitor_config for source_catalog monitor.")
                return
            monitor_source_catalog = import_module("rdmt_spire.monitors.source_catalog")
            self.monitor_objects.append(
                monitor_source_catalog.SourceCatalogMonitor(self.asdf_file, monitor_config["source_catalog"]))

        elif monitor_name == "photoastro":
            config_dirs=["RDMT_SPIRE_LDATA_DIR", "RDMT_SPIRE_RDATA_DIR", "RDMT_SPIRE_L4_DIR"]
            if monitor_config is None or "photoastro" not in monitor_config or len(set(config_dirs) - set(monitor_config["photoastro"])) != 0:
                logger.error("MonitorManager: Missing required config_dirs in monitor_config for photoastro monitor.")
                self.statusCode = StatusCodes.FAILURE
                self.errors.append("MonitorManager: Missing required config_dirs in monitor_config for photoastro monitor.")
                return
            monitor_photoastro = import_module("rdmt_spire.monitors.photoastro")
            self.monitor_objects.append(
                monitor_photoastro.PhotoAstroMonitor(self.asdf_file, monitor_config["photoastro"]))

        elif monitor_name == "base_monitor":
            self.monitor_objects.append(
                BaseMonitor(self.asdf_file)
            )

        else:
            self.statusCode = StatusCodes.FAILURE
            self.errors.append(f"MonitorManager: Monitor not found: {monitor_name}")

    def print_logs(self):
        """
        Prints logs and errors helpful for debugging
        """
        print("Logs:")
        for item in self.log:
            print("    ", item)

        print("Errors:")
        for item in self.errors:
            print("   ", item)

    def process(self):
        """
        The monitors are run and their results are archived.
        """
        try:
            self.log.append(
                f"Monitor manager: processing {len(self.monitor_objects)} monitors"
            )
            for monitor_obj in self.monitor_objects:
                monitor_obj.check_input()
                monitor_obj.run()

        except RuntimeError as e:
            self.statusCode = StatusCodes.FAILURE
            self.errors.append(str(e))

        except Exception as e:
            self.statusCode = StatusCodes.FAILURE
            self.errors.append(f"MonitorManager.process(): failure {e}")

    def archive(self, session, filename, reprocess_number, results_table_class):
        """
        Store aggregated monitor results into a results table row and flush to the database.

        This method looks up (or creates) a results row identified by the composite
        primary key ``(filename, reprocess_number)`` in ``results_table_class``. It then
        iterates over all monitor objects attached to this instance, pulls each monitor's
        data cards (via ``get_data_card('all')``), and assigns their values to matching
        columns on the results row. Finally, it calls ``session.flush()`` to push the
        pending changes to the database transaction's staging area.

        Parameters
        ----------
        session : sqlalchemy.orm.Session
            An active SQLAlchemy session used to retrieve or create the results row
            and to flush in-memory changes.
        filename : str
            The identifier (first component of the composite primary key) for the
            results row to upsert.
        reprocess_number : int
            The reprocessing iteration (second component of the composite primary key)
            for the results row to upsert.
        results_table_class : Type[DeclarativeBase]
            The SQLAlchemy declarative model class representing the results table.
            It must define a composite primary key over ``(filename, reprocess_number)``
            and columns whose names correspond to the monitor data card names.

        Returns
        -------
        None
            This method performs side effects on the database session and does not
            return a value.

        """
        primary_keys = (filename, reprocess_number)
        results_row = session.get(results_table_class, primary_keys)
        if results_row is None:
            logger.info('No existing row found. Making a new row.')
            results_row = results_table_class(filename=filename, reprocess_number=reprocess_number)

        col_names = results_table_class.__table__.c.keys()
        for monitor_obj in self.monitor_objects:
            for data_card in monitor_obj.get_data_card('all'):
                logger.info(f'Adding {data_card.data_name} data + eval.')
                if data_card.data_name in col_names:
                    setattr(results_row, data_card.data_name, data_card.data_value)
                    setattr(results_row, f'{data_card.data_name}_eval', data_card.evaluation_value)
                else:
                    raise ValueError(f"metric ('{data_card.data_name}') does not have a matching column in results table ('L2ScienceResultsTable').")
        
        # flush the changes to the staging area to ensure correct
        logger.info('Flushing changes to database.')
        session.merge(results_row)
        session.flush()

    def response_for_lambda(self) -> Dict[str, int | List[Dict[Any, Any]]]:
        """
        A response object suitable to be used as return value by the lambda handler function.
        The lambda handler can return a value, which must be JSON serializable.
        """
        body_list = []
        if self.statusCode == StatusCodes.SUCCESS:
            for monitor_obj in self.monitor_objects:
                body_list.extend(monitor_obj.get_data_card("all", serialized=True))
        else:
            for error in self.errors:
                body_list.append({"error_message": error})

        response: Dict[str, int | List[Dict[Any, Any]]] = {
            "statusCode": self.statusCode,
            "body": json.dumps(body_list),
        }

        return response
