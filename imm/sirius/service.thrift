#
#
#
# The ImageMatchingService is running on the server to accept image queries
# and return the name that best matches in the pre-trained database  
service ImageMatchingService {
	string send_request(1:binary img_query),
	void ping();
}
