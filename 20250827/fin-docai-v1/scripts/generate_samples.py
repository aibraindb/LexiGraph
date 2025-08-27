
import os, random, datetime, json
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

BASE = os.path.dirname(os.path.dirname(__file__))
OUT_PDF = os.path.join(BASE, "data", "samples")
OUT_GT  = os.path.join(BASE, "data", "labels")
os.makedirs(OUT_PDF, exist_ok=True); os.makedirs(OUT_GT, exist_ok=True)

def gen_invoice_pdf(p_pdf, p_json):
    c = canvas.Canvas(p_pdf, pagesize=LETTER); W,H = LETTER
    c.setFont("Helvetica-Bold",18); c.drawString(1*inch, H-1*inch, "INVOICE")
    c.setFont("Helvetica",10)
    inv_no   = f"INV-{random.randint(1000,9999)}"
    inv_date = datetime.date.today().strftime("%m/%d/%Y")
    due_date = (datetime.date.today()+datetime.timedelta(days=30)).strftime("%m/%d/%Y")
    c.drawString(1*inch, H-1.3*inch, f"Invoice Number: {inv_no}")
    c.drawString(1*inch, H-1.5*inch, f"Invoice Date: {inv_date}")
    c.drawString(1*inch, H-1.7*inch, f"Due Date: {due_date}")
    y = H-2.2*inch
    c.setFont("Helvetica-Bold",10); c.drawString(1*inch, y, "Description")
    c.drawString(3.5*inch, y, "Qty"); c.drawString(4.5*inch, y, "Price"); c.drawString(5.5*inch, y, "Amount")
    c.setFont("Helvetica",10); y -= 0.2*inch
    items=[]; total=0.0
    for i in range(random.randint(3,5)):
        desc=f"Service {i+1}"; qty=random.randint(1,3); price=round(random.uniform(50,200),2); amt=round(qty*price,2)
        items.append({"description":desc,"qty":qty,"price":price,"amount":amt}); total+=amt
        c.drawString(1*inch,y,desc); c.drawRightString(4.0*inch,y,str(qty))
        c.drawRightString(5.3*inch,y,f"{price:.2f}"); c.drawRightString(6.5*inch,y,f"{amt:.2f}"); y-=0.2*inch
    y-=0.2*inch; c.setFont("Helvetica-Bold",12); c.drawRightString(5.3*inch,y,"Total:"); c.drawRightString(6.5*inch,y,f"{total:.2f}")
    c.showPage(); c.save()
    json.dump({"type":"invoice","invoice_number":inv_no,"invoice_date":inv_date,"due_date":due_date,"items":items,"total":round(total,2)}, open(p_json,"w"), indent=2)

def gen_statement_pdf(p_pdf, p_json):
    c = canvas.Canvas(p_pdf, pagesize=LETTER); W,H = LETTER
    c.setFont("Helvetica-Bold",16); c.drawString(1*inch, H-1*inch, "MONTHLY STATEMENT")
    c.setFont("Helvetica",10); acct=f"ACCT-{random.randint(100000,999999)}"; c.drawString(1*inch,H-1.3*inch,f"Account Number: {acct}")
    opening=round(random.uniform(1000,5000),2); c.drawString(1*inch, H-1.5*inch, f"Opening Balance: {opening:.2f}")
    y = H-2.0*inch; c.setFont("Helvetica-Bold",10)
    c.drawString(1*inch,y,"Date"); c.drawString(2*inch,y,"Description"); c.drawRightString(5.5*inch,y,"Amount"); c.drawRightString(7*inch,y,"Balance")
    c.setFont("Helvetica",10); y -= 0.2*inch
    bal=opening; tx=[]
    for i in range(8):
        d=(datetime.date.today()-datetime.timedelta(days=30-i*3)).strftime("%m/%d/%Y")
        desc=random.choice(["POS Purchase","ATM Withdrawal","Salary","Online Transfer"])
        amt=round(random.uniform(-300,300),2); bal=round(bal+amt,2)
        tx.append({"date":d,"description":desc,"amount":amt,"balance":bal})
        c.drawString(1*inch,y,d); c.drawString(2*inch,y,desc)
        c.drawRightString(5.5*inch,y,f"{amt:.2f}"); c.drawRightString(7*inch,y,f"{bal:.2f}"); y-=0.22*inch
    c.setFont("Helvetica-Bold",12); c.drawString(1*inch, y-0.3*inch, f"Closing Balance: {bal:.2f}")
    c.showPage(); c.save()
    json.dump({"type":"bank_statement","account":acct,"opening":opening,"closing":bal,"transactions":tx}, open(p_json,"w"), indent=2)

if __name__=="__main__":
    os.makedirs(OUT_PDF, exist_ok=True); os.makedirs(OUT_GT, exist_ok=True)
    gen_invoice_pdf(os.path.join(OUT_PDF,"sample_invoice.pdf"), os.path.join(OUT_GT,"sample_invoice.json"))
    gen_statement_pdf(os.path.join(OUT_PDF,"sample_statement.pdf"), os.path.join(OUT_GT,"sample_statement.json"))
    print("Generated samples in", OUT_PDF)
