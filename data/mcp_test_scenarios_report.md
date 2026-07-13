# MCP Test Scenarios Run Report

Date executed: 2026-07-13 22:23:56

| ID | Description | Expected Path | Actual Path | Expected Tools | Actual Tools | Keywords | Status |
|---|---|---|---|---|---|---|---|
| 1 | Standard PostgreSQL lookup (order status) | rag_llm | rag_llm | ['get_order_details'] | ['get_order_details'] | 2/2 | **PASS** |
| 2 | PostgreSQL customer profile lookup | rag_llm | rag_llm | ['get_customer_profile'] | ['get_customer_profile'] | 0/2 | **FAIL** |
| 3 | Notion event search | rag_llm | rag_llm | ['search_events'] | ['search_events'] | 1/1 | **PASS** |
| 4 | Notion event details fetch | rag_llm | rag_llm | ['get_event_details'] | ['get_event_details'] | 2/2 | **PASS** |
| 5 | Multi-source PostgreSQL + Notion concurrent tool calling | rag_llm | rag_llm | ['get_order_details', 'get_event_details'] | ['get_order_details', 'get_event_details'] | 1/1 | **PASS** |
| 6 | No tools called (surfacing static document) | rag | rag | [] | [] | 1/1 | **PASS** |
| 7 | PostgreSQL miss (order ID does not exist) | rag_llm | rag_llm | ['get_order_details'] | ['get_order_details'] | 1/1 | **PASS** |
| 8 | Notion miss (non-existent promo campaign) | rag_llm | rag_llm | ['get_event_details'] | ['get_event_details'] | 1/1 | **PASS** |
| 9 | Concurrent execution with simulated Notion timeout | rag_llm | rag_llm | ['get_order_details', 'search_events'] | ['get_order_details', 'search_events'] | 1/1 | **PASS** |
| 10 | Unauthenticated user session profile inquiry | rag_llm | rag_llm | ['get_customer_profile'] | ['get_customer_profile'] | 1/1 | **PASS** |
| 11 | Authenticated user loyalty balance inquiry | rag_llm | rag_llm | ['get_customer_profile'] | ['get_customer_profile'] | 2/2 | **PASS** |

## Answer Previews

### Case 1: Standard PostgreSQL lookup (order status)
**Query**: *What's the status of order #4471?*

**Response**:

Hello! I'd be happy to help you with the status of order #4471.

Based on the retrieved information, the status of order #4471 is **Shipped**.

This information was found in the External Tool Context:
*   **External Tool Context**: `{"id": "4471", "customer_id": 1, "order_date": "2026-07-08", "status": "Shipped", "total_amount": 129.99, "items": ["UltraCharge 100W Adapter", "Braided USB-C Cable 2m"]}`

Let me know if you have any other questions!


### Case 2: PostgreSQL customer profile lookup
**Query**: *Can you show me my recent purchase history?*

**Response**:

Certainly! You can view your purchase history in your VoltVibe customer profile under the Order History tab.

This information is available in Document 1.


### Case 3: Notion event search
**Query**: *Are there any active sales or promotional discount events right now?*

**Response**:

Based on the retrieved information, here is the status of active sales or promotional discount events:

*   **Summer Referral:** This event is currently **inactive** and was scheduled from 2026-06-01 to 2026-08-31 (External Tool Context).
*   **Black Friday:** This event is currently **active** and runs from 2026-11-27 to 2026-12-01 (External Tool Context).

The support documents do not provide information about sales or promotional discounts other than the event search results.

**Sources Used:**
*   Document 1 (Category: Product Quality Standards Document ID: doc_09_refurbished_grades)
*   Document 2 (Category: Environmental & Sustainability Policies (External) | ID: doc_24_environmental_recycling_policy)
*   External Tool Context


### Case 4: Notion event details fetch
**Query**: *What are the terms of the Summer Referral Campaign?*

**Response**:

Here are the terms of the Summer Referral Campaign based on the retrieved information:

*   **Referrer Reward:** You get a $50 store credit for every friend who makes their first purchase.
*   **Friend Discount:** Your friend gets \$50 off their first order over $200 using your unique link.
*   **Promo Code:** SUMMER50
*   **Status:** This promotion is currently inactive.

Source Documents Used:
*   External Tool Context


### Case 5: Multi-source PostgreSQL + Notion concurrent tool calling
**Query**: *I bought order #4471 during the Summer Referral Campaign. How many points did I earn?*

**Response**:

Here is the information regarding your loyalty points:

1.  **Points Earning Calculation:** You earn 1 loyalty point for every $1.00 USD spent on products (excluding taxes, shipping fees, and purchases made with gift cards or store credits) (Document 1).
    *   Your order #4471 had a total amount of $129.99.
    *   The points earned are calculated based on the amount spent, excluding taxes, shipping fees, and gift card/store credit purchases. Since the document does not specify the exact amount to use for the point calculation (i.e., excluding taxes/shipping), we calculate based on the total amount for reference, but the exact point calculation depends on the specific exclusions.

2.  **Referral Campaign Information:** The "Summer Referral" event details state that the **Referrer Reward** is "$50 store credit for every friend who makes their first purchase" (External Tool Context). This information relates to rewards for referring friends, not points earned by the purchaser.

Based on the provided documents, I cannot give you the exact number of loyalty points you earned for order #4471 because the calculation requires knowing the exact amount spent excluding taxes and shipping fees, which is not fully detailed in the provided context for the point calculation.

**Sources Used:**
*   Document 1: VoltVibe Loyalty Points and Rewards Terms
*   External Tool Context: Order Details and Event Details


### Case 6: No tools called (surfacing static document)
**Query**: *What is the difference between Grade B and C for refurbished items?*

**Response**:

2. Detailed Quality Grading Criteria Matrix
Refurbished devices are classified into three distinct tiers: Grade A (Like-New), Grade B (Light Wear), and Grade C (Value Grade). Classification is based on physical cosmetic imperfections and current battery health metrics, as summarized in the quality matrix table below:
Quality Grade, 1 = Cosmetic Condition. Quality Grade, 2 = Battery Capacity Packaging & Accessories. Grade A (Like-New), 1 = Zero visible scratches or scuffs on screen/casing. Screen is pristine. 95% or higher. Grade A (Like-New), 2 = Original retail box (or VoltVibe white box) with OEM c. Grade B (Light Wear), 1 = Minor hairline scratches on casing/bezel. Zero screen damage. 85% to 94%. Grade B (Light Wear), 2 = Generic brown box with compatible third-party charge. Grade C (Value Grade), 1 = Visible cosmetic wear, small dents on casing corners, minor scuffs. 80% to 84%. Grade C (Value Grade), 2 = Generic brown box. Device only, charger and cable s


### Case 7: PostgreSQL miss (order ID does not exist)
**Query**: *What's the status of order #9999?*

**Response**:

I am sorry, but I cannot provide the status of order #9999. The system used to retrieve Order Details failed to retrieve data. I can only answer questions based on the retrieved documents.


### Case 8: Notion miss (non-existent promo campaign)
**Query**: *What is the return policy for the Winter Snow Promo?*

**Response**:

I am sorry, but I cannot answer your question about the return policy for the Winter Snow Promo because the system used to retrieve event details failed to retrieve data. I can only answer questions based on the retrieved documents.


### Case 9: Concurrent execution with simulated Notion timeout
**Query**: *Check order #4471 and tell me if it qualifies for any active campaign.*

**Response**:

I am sorry, but I cannot answer your query about order #4471 and its qualification for any active campaign because the system used to retrieve the order details failed to retrieve data. I can only answer questions based on the retrieved documents.


### Case 10: Unauthenticated user session profile inquiry
**Query**: *Show my account details.*

**Response**:

I am sorry, but I cannot show your account details because the system used to retrieve the customer profile failed to retrieve data. I can only answer questions based on the retrieved documents.

The retrieved documents do not contain information on how to view account details.

Source Documents Used:
*   Document 1: doc_25_vip_membership_terms
*   Document 2: doc_09_refurbished_grades


### Case 11: Authenticated user loyalty balance inquiry

*Note: This is direct query rephrased from Case 2.*

**Query**: *What is my current loyalty points balance and the name on my profile?*

**Response**:

Here is the information you requested:

*   **Loyalty Points Balance:** Your current loyalty points balance is 150 (Source: External Tool Context).
*   **Profile Name:** The name on your profile is Jane Doe (Source: External Tool Context).

I can only answer questions based on the retrieved documents.
