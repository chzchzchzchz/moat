import Foundation

let url = URL(string: "http://127.0.0.1:8000/reflect")!
var request = URLRequest(url: url)
request.httpMethod = "POST"
request.setValue("application/json", forHTTPHeaderField: "Content-Type")

let payload: [String: String] = ["entry": "Test"]
request.httpBody = try? JSONEncoder().encode(payload)

let group = DispatchGroup()
group.enter()

URLSession.shared.dataTask(with: request) { data, response, error in
    if let error = error {
        print("Error: \(error)")
    }
    if let data = data {
        print("Response: \(String(data: data, encoding: .utf8) ?? "")")
    }
    group.leave()
}.resume()

group.wait()
