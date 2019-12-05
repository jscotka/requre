# MIT License
#
# Copyright (c) 2018-2019 Red Hat, Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import inspect
import logging
import os
import time
from _collections_abc import Hashable
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import yaml

from .constants import (
    VERSION_REQURE_FILE,
    ENV_STORAGE_FILE,
    KEY_MINIMAL_MATCH,
    METATADA_KEY,
)
from .exceptions import ItemNotInStorage, StorageNoResponseLeft, StorageKeyError
from .singleton import SingletonMeta
from .utils import StorageMode

# use this sleep to avoid decorating original time function used internally
original_sleep = time.sleep
original_time = time.time

logger = logging.getLogger(__name__)


class StorageKeysInspect:
    @staticmethod
    def get_base_keys(func: Callable) -> List[Any]:
        raise NotImplementedError("Use child classes")


class StorageKeysInspectFull(StorageKeysInspect):
    @staticmethod
    def get_base_keys(func: Callable) -> List[Any]:
        output: List[str] = list()
        # callers module list, to be able to separate requests for various services in one file
        caller_list: List[str] = list()
        for currnetframe in inspect.stack():
            if not inspect.getmodule(currnetframe[0]):
                continue
            module_name = inspect.getmodule(currnetframe[0]).__name__
            if module_name.startswith("_"):
                break
            # avoid to store requre.storage to module stack
            # backward compatibility issue
            if module_name.startswith("requre.storage"):
                continue
            if len(caller_list) and caller_list[-1] == module_name:
                continue
            else:
                caller_list.append(module_name)
        output += caller_list[::-1]
        # module name where function is
        output.append(inspect.getmodule(func).__name__)
        # name of function what were used
        output.append(func.__name__)
        return output


class StorageKeysInspectSimple(StorageKeysInspect):
    @staticmethod
    def get_base_keys(func: Callable) -> List[Any]:
        return [inspect.getmodule(func).__name__, func.__name__]


StorageKeysInspectDefault = StorageKeysInspectFull


class DataStructure:
    """
    Object model for storing data to persistent storage
    """

    OUTPUT_KEY = "output"
    METADATA_KEY = "metadata"

    def __init__(self, output: Any):
        self.output = output
        self._metadata: dict = dict()

    @property
    def metadata(self):
        """
        Medatata for item in storage
        """
        return self._metadata

    @metadata.setter
    def metadata(self, values: dict):
        for k, v in values.items():
            self._metadata[k] = v

    def dump(self):
        """
        Dump object to serializable format

        :return dict
        """
        return {self.METADATA_KEY: self.metadata, self.OUTPUT_KEY: self.output}

    @classmethod
    def create_from_value(cls, dict_repr: dict):
        """
        Create Object representation from dict

        :return DataStructure
        """
        data = cls(dict_repr[cls.OUTPUT_KEY])
        data.metadata = dict_repr[cls.METADATA_KEY]
        return data

    @classmethod
    def create_from_dict(cls, dict_repr: dict):
        """
        Create Object representation from dict

        :return DataStructure
        """
        if DataMiner().key not in dict_repr:
            raise ItemNotInStorage(
                f"Key '{DataMiner().key}' not in the response file. "
                f"(Be sure that you generate the response file "
                f"for this variant.)"
            )
        value = dict_repr[DataMiner().key]
        data = cls(value[cls.OUTPUT_KEY])
        data.metadata = value[cls.METADATA_KEY]
        return data

    @classmethod
    def create_from_dict_with_list(cls, dict_repr: dict):
        """
        Create Object representation from dict with list

        :return DataStructure
        """
        if DataMiner().key not in dict_repr:
            raise ItemNotInStorage(
                f"Key '{DataMiner().key}' not in the response file. "
                f"(Be sure that you generate the response file "
                f"for this variant.)"
            )
        value = dict_repr[DataMiner().key]
        return cls.create_from_list(list_repr=value)

    @classmethod
    def create_from_list(cls, list_repr: list):
        if len(list_repr) == 0:
            raise StorageNoResponseLeft(
                "No responses left. Try to regenerate response file "
                f"({PersistentObjectStorage().storage_file})."
            )
        if PersistentObjectStorage().storage_file_version <= 1:
            # Backward compatibility for requre before using DataStructure
            data = DataStructure(list_repr[0])
        else:
            data = DataStructure.create_from_value(list_repr[0])
        del list_repr[0]
        return data


class DataTypes(Enum):
    List = 1
    Value = 2
    Dict = 3
    DictWithList = 4


class DataMiner(metaclass=SingletonMeta):
    """
    Intermediate class used for proper formatting and keeping backward
    compatibility with requre storage version files
    """

    def __init__(self):
        self.current_time = original_time()
        self.data: Optional[DataStructure] = None
        self.data_type: DataTypes = DataTypes.List
        self.use_latency = False
        self.LATENCY_KEY = "latency"
        self.key: str = "all"
        self.key_stategy_cls = StorageKeysInspectDefault
        self.store_arg_debug_metadata = False
        self.METADATA_ARG_DEBUG_KEY = "log_call_function"
        self.METADATA_CALLER_LIST = "module_call_list"
        self.read_key_exact = False

    def get_latency(self, regenerate=True) -> float:
        """
        Returns latency from last store event

        :param regenerate: if False, then it will not be changed, just displayed
        :return: float
        """
        current_time = original_time()
        diff = current_time - self.current_time
        if regenerate:
            self.current_time = current_time
        return diff

    def dump(self, level, key, values, metadata: Dict):
        """
        Store values in proper format to level[key] item

        :param level: parent dict object
        :param key: item in dict
        :param values: what to store (return value of function)
        :param metadata: store metadata to object from upper level
        :return: None
        """
        ds = DataStructure(values)
        if metadata:
            ds.metadata = metadata
        if self.LATENCY_KEY not in ds.metadata:
            ds.metadata = {self.LATENCY_KEY: self.get_latency()}
        self.data = ds
        item = ds.dump()
        if self.data_type == DataTypes.Dict:
            level.setdefault(key, {})
            level[key][self.key] = item
        elif self.data_type == DataTypes.DictWithList:
            level.setdefault(key, {})
            level[key].setdefault(self.key, [])
            level[key][self.key].append(item)
        elif self.data_type in [DataTypes.List, DataTypes.DictWithList]:
            if PersistentObjectStorage().storage_file_version <= 1:
                # Backward compatibility for requre before using DataStructure
                item = values
            level.setdefault(key, [])
            level[key].append(item)
        elif self.data_type == DataTypes.Dict:
            level.setdefault(key, {})
            level[key][self.key] = item
        else:
            level[key] = item

    def load(self, level):
        """
        Get data from storage_object and trasform it to DataStructure object.

        :param level: where data are stored
        :return: output of function
        """
        data = None
        if self.data_type == DataTypes.List:
            data = DataStructure.create_from_list(level)
        elif self.data_type == DataTypes.Value:
            data = DataStructure.create_from_value(level)
        elif self.data_type == DataTypes.Dict:
            data = DataStructure.create_from_dict(level)
        elif self.data_type == DataTypes.DictWithList:
            data = DataStructure.create_from_dict_with_list(level)
        self.data = data
        if self.use_latency:
            original_sleep(data.metadata.get(self.LATENCY_KEY, 0))
        return data.output

    @property
    def metadata(self):
        return self.data.metadata

    @metadata.setter
    def metadata(self, value):
        self.data.metadata = value


class PersistentObjectStorage(metaclass=SingletonMeta):
    """
    Class implements reading/writing simple JSON requests to dict structure
    and return values based on keys.
    It contains methods to reads/stores data to object and load and store them to YAML file

    storage_object: dict with structured data based on keys (eg. simple JSON requests)
    storage_file: file for reading and writing data in storage_object
    """

    internal_object_key = METATADA_KEY
    version_key = "version_storage_file"
    key_inspect_strategy_key = "key_strategy"

    def __init__(self) -> None:
        # call dump() after store() is called
        self.dump_after_store = False
        self.mode: StorageMode = StorageMode.default
        self.is_flushed = True
        self.storage_object: dict = {}
        self._storage_file: Optional[str] = None
        storage_file_from_env = os.getenv(ENV_STORAGE_FILE)
        if storage_file_from_env:
            self.storage_file = storage_file_from_env

    @property
    def metadata(self):
        """
        Placeholder to store some custom configuration, or store custom data,
        eg. version of storage file, or some versions of packages

        :return: dict
        """
        return self.storage_object.get(self.internal_object_key, {})

    @metadata.setter
    def metadata(self, key_dict: dict):
        if self.internal_object_key not in self.storage_object:
            self.storage_object[self.internal_object_key] = {}
        for k, v in key_dict.items():
            self.storage_object[self.internal_object_key][k] = v
        self.is_flushed = False

    @property
    def storage_file_version(self):
        """
        Get version of persistent storage file
        :return: int
        """
        return self.metadata.get(self.version_key, 0)

    def do_store(self, keys):

        if self.mode == StorageMode.read:
            return False

        elif self.mode == StorageMode.write:
            return True

        elif self.mode == StorageMode.append:
            if DataMiner().data_type in [DataTypes.DictWithList, DataTypes.List]:
                # append possible
                return True

            if (
                DataMiner().data_type in [DataTypes.Dict, DataTypes.Value]
                and keys not in self
            ):
                # keys present => we can read the values
                return True
            return False

        key_present = keys in self
        logger.warning(
            f"No storage-mode set. Checking presence of the keys: {key_present} "
            f"=> {'read' if key_present else 'store'}"
        )
        return not key_present

    @storage_file_version.setter  # type: ignore
    def storage_file_version(self, value: int):
        self.metadata = {self.version_key: value}

    def _set_storage_metadata_if_not_set(self):
        """
        Set storage file version if not already set to latest version.
        """
        if self.metadata.get(self.version_key) is None:
            self.storage_file_version = VERSION_REQURE_FILE
        if self.metadata.get(self.key_inspect_strategy_key) is None:
            self.metadata = {
                self.key_inspect_strategy_key: DataMiner().key_stategy_cls.__name__
            }
        data_type_name = DataTypes.__name__
        if self.metadata.get(data_type_name) is None:
            self.metadata = {data_type_name: DataMiner().data_type.value}

    @property
    def storage_file(self):
        return self._storage_file

    @storage_file.setter
    def storage_file(self, value):
        self._storage_file = value

        if not os.path.exists(self._storage_file):
            if self.mode == StorageMode.default:
                self.mode = StorageMode.write
            self.is_flushed = False
            self.storage_object = {}
        else:
            if self.mode == StorageMode.default:
                self.mode = StorageMode.read
            self.storage_object = self.load()

    @staticmethod
    def transform_hashable(keys: List) -> List:
        output: List = []
        for item in keys:
            if not item:
                output.append("empty")
            elif not isinstance(item, Hashable):
                output.append(str(item))
            else:
                output.append(item)
        return output

    def store(self, keys: List, values: Any, metadata: Dict) -> None:
        """
        Stores data to dictionary object based on keys values it will create structure
        if structure does not exist

        It implicitly changes type to string if key is not hashable

        :param keys: items what will be used as keys for dictionary
        :param values: It could be whatever type what is used in original object handling
        :return: None
        """
        self._set_storage_metadata_if_not_set()
        current_level = self.storage_object
        hashable_keys = self.transform_hashable(keys)
        for item_num in range(len(hashable_keys)):
            item = hashable_keys[item_num]
            if item_num + 1 < len(hashable_keys):
                if not current_level.get(item):
                    current_level[item] = {}
            else:
                DataMiner().dump(
                    level=current_level, key=item, values=values, metadata=metadata
                )
            current_level = current_level[item]
        self.is_flushed = False

        if self.dump_after_store:
            self.dump()
        logger.debug(f"Storing response to: {self.storage_file}: {hashable_keys}")

    def read(self, keys: List) -> Any:
        """
        Reads data from dictionary object structure based on keys.
        If keys does not exists

        It implicitly changes type to string if key is not hashable

        :param keys: key list for searching in dict
        :return: value assigged to key items
        """

        hashable_keys = self.transform_hashable(keys)
        matcher = DictMatch(self.storage_object)
        if DataMiner().read_key_exact:
            current_level = matcher.match_exact(hashable_keys)
        else:
            current_level = matcher.longest_match(hashable_keys)

        return DataMiner().load(level=current_level)

    def __contains__(self, item) -> bool:
        current_level = self.storage_object
        hashable_keys = self.transform_hashable(item)
        for item in hashable_keys:
            if item not in current_level:
                return False
            current_level = current_level[item]
        if DataMiner().data_type in [DataTypes.Dict, DataTypes.DictWithList]:
            return DataMiner().key in current_level
        return True

    def __getitem__(self, key):
        return self.read(keys=key)

    def __setitem__(self, key, value):
        self.store(keys=key, values=value, metadata={})

    def __delitem__(self, key):
        current_level = self.storage_object
        hashable_keys = self.transform_hashable(key)
        last_level = None
        for item in hashable_keys:
            if item not in current_level:
                raise ItemNotInStorage(
                    f"Key not found in the persistent storage: {key}"
                )
            last_level = current_level
            current_level = current_level[item]

        if DataMiner().data_type == DataTypes.Dict:
            del current_level[DataMiner().key]
        else:
            del last_level[key[-1]]

    def dump(self) -> None:
        """
        Explicitly stores content of storage_object to storage_file path

        This method is also called when object is deleted and is set write mode to True

        :return: None
        """
        if self.mode in [StorageMode.write, StorageMode.append]:
            self._set_storage_metadata_if_not_set()
            if self.is_flushed:
                return None
            with open(self.storage_file, "w") as yaml_file:
                yaml.dump(self.storage_object, yaml_file, default_flow_style=False)
            self.is_flushed = True

    def load(self) -> Dict:
        """
        Explicitly loads file content of storage_file to storage_object and return as well

        :return: dict
        """
        with open(self.storage_file, "r") as yaml_file:
            output = yaml.safe_load(yaml_file)
        self.storage_object = output
        # set proper storage strategy if stored in file
        if self.metadata.get(self.key_inspect_strategy_key):
            logger.debug(
                f"Used key strategy: {self.metadata.get(self.key_inspect_strategy_key)}"
            )
            DataMiner().key_stategy_cls = globals()[
                self.metadata.get(self.key_inspect_strategy_key)
            ]
        return output


def use_persistent_storage_without_overwriting(cls):
    class ClassWithPersistentStorage(cls):
        persistent_storage: Optional[
            PersistentObjectStorage
        ] = PersistentObjectStorage()

    ClassWithPersistentStorage.__name__ = cls.__name__
    return ClassWithPersistentStorage


class StorageCounter:
    counter = 0
    dir_suffix = "file_storage"
    previous = None

    @classmethod
    def reset_counter_if_changed(cls):
        current = os.path.basename(PersistentObjectStorage().storage_file)
        if cls.previous != current:
            cls.counter = 0
            cls.previous = current

    @classmethod
    def next(cls):
        cls.counter += 1
        return cls.counter

    @staticmethod
    def storage_file():
        return os.path.basename(PersistentObjectStorage().storage_file)

    @staticmethod
    def storage_dir():
        return os.path.dirname(PersistentObjectStorage().storage_file)


class DictMatch:
    def __init__(self, dict_obj: Dict = None):
        self._dict = (
            PersistentObjectStorage().storage_object if dict_obj is None else dict_obj
        )

    def list_all_items(self, internal_obj: Dict = None, keys: Optional[List] = None):
        keys = keys or []
        tmp_obj = self._dict if internal_obj is None else internal_obj
        if self.minimal_final_match(
            dict_obj=tmp_obj, metadata=PersistentObjectStorage().metadata
        ):
            yield keys
            return
        if isinstance(tmp_obj, dict):
            for k, v in tmp_obj.items():
                if METATADA_KEY == k:
                    continue
                yield from self.list_all_items(internal_obj=v, keys=keys + [k])

    def match_exact(self, keys: List) -> Any:
        current_level = self._dict
        for key in keys:
            if key not in current_level:
                raise ItemNotInStorage(
                    f"Keys {keys} not in storage:{PersistentObjectStorage().storage_file},"
                    f" there are {list(self.list_all_items())}"
                )
            current_level = current_level[key]
        return current_level

    def longest_match(self, keys: List):
        all_items = list(self.list_all_items())
        for item in sorted(all_items, key=len, reverse=True):
            if (
                len(keys) < KEY_MINIMAL_MATCH
                or len(item) < KEY_MINIMAL_MATCH
                or not list(reversed(keys))[:KEY_MINIMAL_MATCH]
                == list(reversed(item))[:KEY_MINIMAL_MATCH]
            ):
                continue
            item_nonfinal = item[:-KEY_MINIMAL_MATCH]
            keys_nonfinal = keys[:-KEY_MINIMAL_MATCH]
            counter = 0
            matched_keys = []
            debug_keys = []
            print(">", item_nonfinal, keys_nonfinal)
            for key in keys_nonfinal:
                print(f"try: {key} in {item_nonfinal}, {counter}")
                if counter >= len(item_nonfinal):
                    break
                if key == item_nonfinal[counter]:
                    debug_keys.append(key)
                    matched_keys.append(key)
                    print(f"matched {key}")
                    counter += 1
                else:
                    print(f"SKIP {key}")
                    debug_keys.append(f"SKIP {key}")

            print(f"Key match: {debug_keys + keys[-KEY_MINIMAL_MATCH:]}")
            return self.match_exact(matched_keys + keys[-KEY_MINIMAL_MATCH:])
        raise ItemNotInStorage(f"Not able to match keys {keys} in {list(all_items)}")

    @staticmethod
    def get_final_keys(internal_dict: Dict) -> List:
        keys: List = []
        tmp_dict = internal_dict
        for cntr in range(KEY_MINIMAL_MATCH):
            if not isinstance(tmp_dict, dict):
                raise StorageKeyError(
                    f"There is expected Dict object {tmp_dict} in level {cntr} in {internal_dict}"
                )
            if len(tmp_dict.keys()) != 1:
                raise StorageKeyError(
                    f"More than 1 key in level {cntr} in {tmp_dict} in {internal_dict}"
                )
            key = list(tmp_dict.keys())[0]
            keys.append(key)
            tmp_dict = tmp_dict[key]
            print(f"========= {keys}")
        return keys

    @staticmethod
    def minimal_final_match(dict_obj: Dict, metadata: Dict) -> bool:
        """
        minimal match lenght based on DataType information
        :param dict_obj:
        :param metadata:
        :return:
        """
        tmp_dict = dict_obj
        first_item: Dict = {}
        key_name = DataTypes.__name__
        ds = DataTypes.List
        if key_name in metadata:
            ds = DataTypes(metadata.get(key_name))
        if ds == DataTypes.DictWithList:
            if isinstance(tmp_dict, dict):
                tmp_first_item = tmp_dict[list(tmp_dict.keys())[0]]
                if isinstance(tmp_first_item, list):
                    first_item = tmp_first_item[0]
        if ds == DataTypes.Dict:
            if isinstance(tmp_dict, dict):
                first_item = tmp_dict[list(tmp_dict.keys())[0]]
        if ds == DataTypes.List:
            if isinstance(tmp_dict, list) and len(tmp_dict):
                first_item = tmp_dict[0]
        if ds == DataTypes.Value:
            if isinstance(tmp_dict, dict):
                first_item = tmp_dict
        if isinstance(first_item, dict) and DataMiner().LATENCY_KEY in first_item.get(
            DataStructure.METADATA_KEY, {}
        ):
            print(f"Minimal key path ({KEY_MINIMAL_MATCH}) matched")
            return True
        return False
