import json
import logging
from speechless import Editor
from speechless.utils import NULL_LOGGER

logger=logging.getLogger()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        handlers=[logging.StreamHandler()],
                        format='%(asctime)s %(levelname)s %(message)s')

    with open('test.json', 'r') as fp:
        jsonSpecs = json.load(fp)
        editor = Editor.fromJSON(jsonSpecs, logger=logger)
        ranges = Editor.parseJSONRanges(jsonSpecs['ranges'])
        editor.toJSON(ranges, 'test2.json')
        editor.edit('video2.mp4', ranges, 'out1.mp4')
