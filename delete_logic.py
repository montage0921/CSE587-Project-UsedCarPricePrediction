import streamlit as st

def delete_widget():
    with st.form("Delete by key"):
        id=st.number_input("Enter the id of the record",step=1)
        delete_btn=st.form_submit_button("Delete")
    return delete_btn,id

def delete_byId(id, conn, cursor):
    delete_query = """DELETE FROM used_cars WHERE id = %s"""
    try:
        cursor.execute(delete_query, (id,)) # cursor only accepts tuple
        conn.commit()
    except Exception as e:
        st.error(f"Record id({id} cannnot be found in the database)")