import streamlit as st
import pymysql


def edit_widget():
    with st.form("Edit by key"):
        id=st.number_input("Enter the id of the record",step=1)
        edit_btn=st.form_submit_button("Edit")
    return edit_btn,id

@st.cache_data
def get_by_id(id,_cursor):
    query_getById="SELECT * FROM used_cars where id=%s"
    _cursor.execute(query_getById,(id,))
    return _cursor.fetchone()

def render_update_widget(_conn,_cursor,res):

    if res is None:
        st.warning("No record found.")
        return
    
    # unpack the res tuple
    (
        id, year, make, model, price, mileage, miles_per_gallon, transmission, owners, VIN,
        accidents, open_recalls, cylinder, fuel, drive_type, miles_per_gallon_equivalent,
        range_when_new, time_to_fully_charge, is_electric, bed_length,
        exterior_color, interior_color, class1, is_auction, has_odometer_issue,
        certification
    ) = res

    # create seperate dict for general features, gas-only, electric-only features
    general = {
        "id":id,
        "is_electric":is_electric,
        "year": year,
        "make": make,
        "model": model,
        "price": price,
        "mileage": mileage,
        "transmission": transmission,
        "owners": owners,
        "VIN": VIN,
        "accidents": accidents,
        "open_recalls": open_recalls,
        "cylinders": cylinder,
        "drive_type": drive_type,
        "bed_length": bed_length,
        "exterior_color": exterior_color,
        "interior_color": interior_color,
        "class": class1,
        "is_auction": is_auction,
        "has_odometer_issue": has_odometer_issue,
        "certification": certification
    }

    gas_car = {
        "fuel": fuel,
        "miles_per_gallon": miles_per_gallon,
    }

    electric_car = {
        "miles_per_gallon_equivalent": miles_per_gallon_equivalent,
        "range_when_new": range_when_new,
        "time_to_fully_charge": time_to_fully_charge
    }

    # render the update form dynamically based on car type
    if is_electric:
        with st.form("electric_car_update"):
            st.subheader("Update Electric Car Details")
            
            # general
            st.write("**General Information**")
            general["year"] = st.number_input("Year", value=general["year"])
            general["make"] = st.text_input("Make", value=general["make"])
            general["model"] = st.text_input("Model", value=general["model"])
            general["price"] = st.number_input("Price", value=general["price"])
            general["mileage"] = st.number_input("Mileage", value=general["mileage"])
            general["transmission"] = st.text_input("Transmission", value=general["transmission"])
            general["owners"] = st.number_input("Owners", value=general["owners"])
            general["VIN"] = st.text_input("VIN", value=general["VIN"])
            general["accidents"] = st.text_input("Accidents Condition", value=general["accidents"])
            general["open_recalls"] = st.number_input("Number of Open Recalls", value=general["open_recalls"])
            general["cylinders"] = st.number_input("Cylinders", value=general["cylinders"])
            general["drive_type"] = st.text_input("Drive Type", value=general["drive_type"])
            general["bed_length"] = st.number_input("Bed Length", value=general["bed_length"])
            general["exterior_color"] = st.text_input("Exterior Color", value=general["exterior_color"])
            general["interior_color"] = st.text_input("Interior Color", value=general["interior_color"])
            general["class"] = st.text_input("Class", value=general["class"])

            # boolean value options
            general["is_auction"] = st.selectbox(
            "Is auctioned?", options=[False, True], format_func=lambda x: "Yes" if x else "No", index=int(general["is_auction"])
            )
            general["has_odometer_issue"] = st.selectbox(
            "Has Odometer Issue?", options=[False, True], format_func=lambda x: "Yes" if x else "No", index=int(general["has_odometer_issue"])
            )

            general["certification"] = st.text_input("Certification", value=general["certification"])
            
            # Electric car-specific attributes
            st.write("**Electric Car Information**")
            electric_car["miles_per_gallon_equivalent"] = st.text_input("MPG Equivalent", value=electric_car["miles_per_gallon_equivalent"])
            electric_car["range_when_new"] = st.number_input("Range When New", value=electric_car["range_when_new"])
            electric_car["time_to_fully_charge"] = st.number_input(
                "Time to Fully Charge (hrs)", value=electric_car["time_to_fully_charge"]
            )
            
            # Submit button
            submitted = st.form_submit_button("Submit Update")
            if submitted:
                update_electric_car_details(_conn,_cursor,general,electric_car)
    else:
        with st.form("gas_car_update"):
            st.subheader("Update Gas Car Details")
            # General information
            st.write("**General Information**")
            general["year"] = st.number_input("Year", value=general["year"])
            general["make"] = st.text_input("Make", value=general["make"])
            general["model"] = st.text_input("Model", value=general["model"])
            general["price"] = st.number_input("Price", value=general["price"])
            general["mileage"] = st.number_input("Mileage", value=general["mileage"])
            general["transmission"] = st.text_input("Transmission", value=general["transmission"])
            general["owners"] = st.number_input("Owners", value=general["owners"])
            general["VIN"] = st.text_input("VIN", value=general["VIN"])
            general["accidents"] = st.text_input("Accidents Condition", value=general["accidents"])
            general["open_recalls"] = st.number_input("Number of Open Recalls", value=general["open_recalls"])
            general["cylinders"] = st.number_input("Cylinders", value=general["cylinders"])
            general["drive_type"] = st.text_input("Drive Type", value=general["drive_type"])
            general["bed_length"] = st.number_input("Bed Length", value=general["bed_length"])
            general["exterior_color"] = st.text_input("Exterior Color", value=general["exterior_color"])
            general["interior_color"] = st.text_input("Interior Color", value=general["interior_color"])
            general["class"] = st.text_input("Class", value=general["class"])

            # boolean value options
            general["is_auction"] = st.selectbox(
            "Is auctioned?", options=[False, True], format_func=lambda x: "Yes" if x else "No", index=int(general["is_auction"])
            )
            general["has_odometer_issue"] = st.selectbox(
            "Has Odometer Issue?", options=[False, True], format_func=lambda x: "Yes" if x else "No", index=int(general["has_odometer_issue"])
            )

            general["certification"] = st.text_input("Certification", value=general["certification"])


            # Gas car-specific attributes
            st.write("**Gas Car Information**")
            gas_car["fuel"] = st.text_input("Fuel Type", value=gas_car["fuel"])
            gas_car["miles_per_gallon"] = st.text_input("Miles Per Gallon (MPG)", value=gas_car["miles_per_gallon"])
            
            # Submit button
            submitted = st.form_submit_button("Submit Update")
            if submitted:
                update_gas_car_details(_conn,_cursor,general,gas_car)

def update_gas_car_details(_conn,_cursor,general,gas):
    query = """
        UPDATE used_cars
        SET year = %s,
            make = %s,
            model = %s,
            price = %s,
            mileage = %s,
            transmission = %s,
            owners = %s,
            VIN = %s,
            accidents = %s,
            open_recalls = %s,
            cylinders = %s,
            drive_type = %s,
            bed_length = %s,
            exterior_color = %s,
            interior_color = %s,
            class = %s,
            is_auction = %s,
            has_odometer_issue = %s,
            certification = %s,
            fuel=%s,
            miles_per_gallon=%s
        WHERE id = %s
    """

    _cursor.execute(query, (
        general["year"], general["make"], general["model"], general["price"], 
        general["mileage"], general["transmission"], general["owners"], 
        general["VIN"], general["accidents"], general["open_recalls"], 
        general["cylinders"], general["drive_type"], general["bed_length"], 
        general["exterior_color"], general["interior_color"], general["class"], 
        general["is_auction"], general["has_odometer_issue"], general["certification"],
        gas["fuel"],gas["miles_per_gallon"],
        general["id"]  # Include `id` for the WHERE clause
    ))

    
    _conn.commit()
    st.success("The record has been successfully updated")



def update_electric_car_details(_conn, _cursor,general,electric):
    query = """
        UPDATE used_cars
        SET year = %s,
            make = %s,
            model = %s,
            price = %s,
            mileage = %s,
            transmission = %s,
            owners = %s,
            VIN = %s,
            accidents = %s,
            open_recalls = %s,
            cylinders = %s,
            drive_type = %s,
            bed_length = %s,
            exterior_color = %s,
            interior_color = %s,
            class = %s,
            is_auction = %s,
            has_odometer_issue = %s,
            certification = %s,
            fuel=%s,
            miles_per_gallon=%s,
            miles_per_gallon_equivalent=%s,
            range_when_new=%s,
            time_to_fully_charge=%s
        WHERE id = %s
    """

    _cursor.execute(query, (
        general["year"], general["make"], general["model"], general["price"], 
        general["mileage"], general["transmission"], general["owners"], 
        general["VIN"], general["accidents"], general["open_recalls"], 
        general["cylinders"], general["drive_type"], general["bed_length"], 
        general["exterior_color"], general["interior_color"], general["class"], 
        general["is_auction"], general["has_odometer_issue"], general["certification"],
        electric["miles_per_gallon_equivalent"],electric["range_when_new"],electric["time_to_fully_charge"],
        general["id"]  # Include `id` for the WHERE clause
    ))

    _conn.commit()
    st.success("The record has been successfully updated")

