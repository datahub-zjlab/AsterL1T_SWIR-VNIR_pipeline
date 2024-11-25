import psycopg2
from shapely.wkt import dumps, loads
from aster_core.token_config import params

# Function to retrieve files based on spatial, temporal, and cloud cover criteria
def retrieve_files(region, time_start='2000-01-01', time_end='2008-01-01', cloud_cover=101, download_flag=True):
    """
    Parameters:
    region: shapely.geometry.Polygon object or None
    time_start: default is '2001-01-01'
    time_end: default is '2008-01-01'
    cloud_cover: default is 101 (return all files)

    Returns: 
    oss_url, producer_granule_id, ST_AsText(polygon), cloud_cover
    """

    conn = psycopg2.connect(**params)
    cur = conn.cursor()

    if region is not None:
        wkt_region = dumps(region)

    if download_flag:  # Only return data that has been downloaded to OSS
        if region is not None:
            retrieve_query = """
            SELECT oss_url, producer_granule_id, ST_AsText(polygon), cloud_cover
            FROM aster_metadata
            WHERE (CASE 
                WHEN ST_Area(polygon) < ST_Area(ST_ShiftLongitude(polygon)) THEN ST_Intersects(polygon, ST_GeomFromText(%s, 4326))
                ELSE ST_Intersects(ST_ShiftLongitude(polygon), ST_ShiftLongitude(ST_GeomFromText(%s, 4326)))
            END)
            AND time_end > %s
            AND time_end < %s
            AND cloud_cover < %s
            AND granule_size > 90
            AND download_complete = TRUE;
            """
        else:
            retrieve_query = """
            SELECT oss_url, producer_granule_id, ST_AsText(polygon), cloud_cover
            FROM aster_metadata
            WHERE time_end > %s
            AND time_end < %s
            AND cloud_cover < %s
            AND granule_size > 90
            AND download_complete = TRUE;
            """
    else:  # Return all Aster data
        if region is not None:
            retrieve_query = """
            SELECT hdf_link, producer_granule_id, ST_AsText(polygon), cloud_cover
            FROM aster_metadata
            WHERE (CASE 
                WHEN ST_Area(polygon) < ST_Area(ST_ShiftLongitude(polygon)) THEN ST_Intersects(polygon, ST_GeomFromText(%s, 4326))
                ELSE ST_Intersects(ST_ShiftLongitude(polygon), ST_ShiftLongitude(ST_GeomFromText(%s, 4326)))
            END)
            AND time_end > %s
            AND time_end < %s
            AND granule_size > 90
            AND cloud_cover < %s;
            """
        else:
            retrieve_query = """
            SELECT hdf_link, producer_granule_id, ST_AsText(polygon), cloud_cover
            FROM aster_metadata
            WHERE time_end > %s
            AND time_end < %s
            AND granule_size > 90
            AND cloud_cover < %s;
            """

    try:
        # Execute the retrieval operation
        if region is not None:
            cur.execute(retrieve_query, (wkt_region, wkt_region, time_start, time_end, cloud_cover))
        else:
            cur.execute(retrieve_query, (time_start, time_end, cloud_cover))

        # Fetch the query results
        rows = cur.fetchall()
        # Create a dictionary to store the results
        result_dict = {}
        # Store the results in the dictionary
        for row in rows:
            file_url = row[0]  # oss_url(local) or hdf_link(YOU CAN DOWNLOAD FILE USING THIS Link)
            granule_id = row[1]  # producer_granule_id
            polygon = loads(row[2])  # polygon
            cloud_cover = row[3]  # cloud_cover
            # if polygon.area < 20: # remove bad polygons TODO fix bad polygons
            result_dict[granule_id] = {'file_url': file_url, 'polygon': polygon, 'cloud_cover': cloud_cover}

        return result_dict

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        return None

    finally:
        # Close the cursor and connection
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def retrieve_aod_files(region, time_start='2000-01-01', time_end='2008-01-01'):
    """
    Parameters:
    region: shapely.geometry.Polygon object
    request_time: 

    Returns: 
    aod_file, begin_time
    """

    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    
    if not isinstance(region,str):
        wkt_region = dumps(region)
    else:
        wkt_region = region

    # 查询
    if region is not None:
        retrieve_query = """
        SELECT hdf_link, producer_granule_id, ST_AsText(polygon),time_start,time_end
        FROM modis_metadata
        WHERE (CASE 
            WHEN ST_Area(polygon) < ST_Area(ST_ShiftLongitude(polygon)) THEN ST_Intersects(polygon, ST_GeomFromText(%s, 4326))
            ELSE ST_Intersects(ST_ShiftLongitude(polygon), ST_ShiftLongitude(ST_GeomFromText(%s, 4326)))
        END)
        AND time_end > %s
        AND time_end < %s;
        """
    else:
        retrieve_query = """
        SELECT hdf_link, producer_granule_id, ST_AsText(polygon),time_start,time_end
        FROM modis_metadata
        WHERE time_end > %s
        AND time_end < %s;
        """

    try:
        # Execute the retrieval operation
        if region is not None:
            cur.execute(retrieve_query, (wkt_region, wkt_region, time_start, time_end))
        else:
            cur.execute(retrieve_query, (time_start, time_end))

        # Fetch the query results
        rows = cur.fetchall()
        # Create a dictionary to store the results
        result_dict = {}
        # Store the results in the dictionary
        for row in rows:
            file_url = row[0]  # oss_url(local) or hdf_link(YOU CAN DOWNLOAD FILE USING THIS Link)
            granule_id = row[1]  # producer_granule_id
            polygon = loads(row[2])  # polygon
            time_start = row[3]
            time_end = row[4]
            # if polygon.area < 10000: # remove bad polygons TODO fix bad polygons
            result_dict[granule_id] = {'file_url': file_url, 'polygon': polygon, 'time_start':time_start, 'time_end':time_end}

        return result_dict

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        return None

    finally:
        # Close the cursor and connection
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def retrieve_gdem_files(region):
    """
    Parameters:
    region: shapely.geometry.Polygon object
    request_time: 

    Returns: 
    gdem_file
    """

    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    
    if not isinstance(region,str):
        wkt_region = dumps(region)
    else:
        wkt_region = region

    # 查询
    retrieve_query = """
    SELECT producer_granule_id, ST_AsText(polygon)
    FROM gdem_metadata
    WHERE (CASE 
        WHEN ST_Area(polygon) < ST_Area(ST_ShiftLongitude(polygon)) THEN ST_Intersects(polygon, ST_GeomFromText(%s, 4326))
        ELSE ST_Intersects(ST_ShiftLongitude(polygon), ST_ShiftLongitude(ST_GeomFromText(%s, 4326)))
    END);
    """

    try:
        cur.execute(retrieve_query, (wkt_region, wkt_region))
        # Fetch the query results
        rows = cur.fetchall()
        # Create a dictionary to store the results
        result_dict = {}
        # Store the results in the dictionary
        for row in rows:
            granule_id = row[0]  # producer_granule_id
            polygon = loads(row[1])  # polygon
            # if polygon.area < 10000: # remove bad polygons TODO fix bad polygons
            result_dict[granule_id] = {'polygon': polygon}

        return result_dict

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        return None

    finally:
        # Close the cursor and connection
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()