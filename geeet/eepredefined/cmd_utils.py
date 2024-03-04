"""geeet eepredefined command-line utilities"""

def cmd_parser(description):
    """
    Command-line parser
    Read a pre-defined tasks metadata table, 
    filter it, and use it to generate the ee export tasks. 
    """
    import argparse
    p=argparse.ArgumentParser()
    p.description=(description)
    p.add_argument("table", help="Tasks metadata table.",
                   type=argparse.FileType("r"))
    p.add_argument("kwargs_json", 
        help="JSON file with `landsat.mapped_export` keyword arguments",
        type=argparse.FileType("r"))
    p.add_argument("-q","--query", help="Query string to filter table. "+
                   "See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html", 
                   type=str)
    return p


def ctseb():
    """ geeet-ctseb EE tasks generator
    
    usage: geeet-ctseb [-h] [-q QUERY] table kwargs_json
    
    positional arguments:
      table                 Tasks metadata table.
      kwargs_json           JSON file with eereducers.tseb keyword arguments
    
    options:
      -h, --help            show this help message and exit
      -q QUERY, --query QUERY
                            Query string to filter table. See
                            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

    The tasks metadata table should contain the following columns:
        region, date_start, date_end, fileNamePrefix, description

    More details about `kwargs_json` (see `landsat.mapped_export`): 
        - landsat_kwargs: Keyword arguments for `landsat.collection`, except for:
            date_start, date_end, region
        - reducer_kwargs: Keyword arguments for `ee.Reducer.mean`
        - export_kwargs: Keyword arguments for ee.batch.Export.to(Drive | cloudStorage)
        - **kwargs: Keyword arguments for `reducers.image_collection`:
            - feature_properties
            - img_properties
            - mean_bands
            - sum_bands
    """
    import json, ee
    import pandas as pd
    from geeet.eepredefined import landsat
    from geeet.eepredefined import workflows
    p = cmd_parser("geeet-ctseb EE tasks generator")
    cargs = p.parse_args()

    query = cargs.query

    with open(cargs.kwargs_json.name) as j:
        kwargs = json.load(j)

    table = pd.read_csv(cargs.table.name)
    if(query): table = table.query(query)
    if len(table)>3000: raise BaseException("Can't export more than 3000 tasks.")

    ee.Initialize()
    eetasks = ee.data.listOperations() 
    ntasks = 0 
    for task in eetasks:
        state = task["metadata"]["state"]
        if state in ['RUNNING','PENDING']: ntasks+=1

    if (len(table)+ntasks)>3000: raise BaseException(
       f"The total number of tasks to submit and currently PENDING+RUNNING tasks ({ntasks}) exceeds 3000."
       +" Modify the query to reduce the number of tasks to submit, or wait for more tasks to complete."
    )

    print(f"Exporting {len(table)} tasks")

    tseb_kwargs = kwargs.pop("tseb_kwargs")
    workflow_kwargs = kwargs.pop("workflow_kwargs")
    landsat_kwargs = kwargs.pop("landsat_kwargs")
    reducer_kwargs = kwargs.pop("reducer_kwargs")
    export_kwargs = kwargs.pop("export_kwargs")

    export_to = "drive"

    if "bucket" in export_kwargs:
        export_to = "cloudStorage"

    tseb = landsat.tseb_series(**tseb_kwargs)
    masked_et_workflow = workflows.masked_et(**workflow_kwargs)

    workflow = [tseb] + masked_et_workflow

    def ctseb_apply(task):
        """Applys ctseb to one row of the metadata table 
        (starts one EE task)
        """
        landsat_args = {**landsat_kwargs, 
            "date_start": task.date_start,
            "date_end": task.date_end,
            "region": task.region
        } 
        export_args = {**export_kwargs,
            "description": task.description,
            "fileNamePrefix": task.fileNamePrefix,
        }
        return landsat.mapped_export(
            workflow,
            task.region,
            landsat_args,
            reducer_kwargs,
            export_args,
            to=export_to,
            **kwargs
        )

    tasks = table.apply(ctseb_apply, axis=1)
