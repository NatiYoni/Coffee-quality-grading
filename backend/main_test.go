package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func pythonStub(t *testing.T, handler http.HandlerFunc) *gin.Engine {
	t.Helper()
	gin.SetMode(gin.TestMode)
	server := httptest.NewServer(handler)
	originalURL, originalClient := pythonServiceURL, pythonClient
	pythonServiceURL = server.URL
	t.Cleanup(func() {
		server.Close()
		pythonServiceURL, pythonClient = originalURL, originalClient
	})
	return setupRouter()
}

func postJSON(router http.Handler, path, body string) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(recorder, request)
	return recorder
}

func TestExplainForwardsFeaturesWithoutLLMCredentials(t *testing.T) {
	router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/explain" || r.Method != http.MethodPost {
			t.Errorf("unexpected upstream route: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "" {
			t.Error("the Go proxy must not handle Gemini credentials")
		}
		var body PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Features["Aroma"] != 8.2 {
			t.Errorf("features changed: %#v", body.Features)
		}
		w.Write([]byte(`{"method":"validated_evidence_plan","sentences":[]}`))
	})
	response := postJSON(router, "/api/explain", `{"features":{"Aroma":8.2}}`)
	if response.Code != 200 || response.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("unexpected response: %d %#v", response.Code, response.Header())
	}
}

func TestPredictionProxiesPreserveErrorsAndRetryAfter(t *testing.T) {
	for _, endpoint := range []string{"/predict", "/explain", "/predict-roast", "/predict-defect"} {
		for _, status := range []int{400, 422, 429, 502, 503, 504} {
			t.Run(endpoint+"/"+http.StatusText(status), func(t *testing.T) {
				router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path != endpoint {
						t.Errorf("path = %q; want %q", r.URL.Path, endpoint)
					}
					w.Header().Set("Retry-After", "60")
					w.WriteHeader(status)
					w.Write([]byte(`{"detail":{"code":"test_error","message":"Try later."}}`))
				})
				response := postJSON(router, "/api"+endpoint, `{"features":{}}`)
				if response.Code != status || response.Header().Get("Retry-After") != "60" {
					t.Fatalf("error lost: status=%d headers=%v", response.Code, response.Header())
				}
				if !strings.Contains(response.Body.String(), "test_error") {
					t.Fatal("structured error body lost")
				}
			})
		}
	}
}

func TestMalformedRequestNeverCallsUpstream(t *testing.T) {
	router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("malformed input reached upstream")
	})
	if response := postJSON(router, "/api/explain", `{bad`); response.Code != 400 {
		t.Fatalf("status = %d; want 400", response.Code)
	}
}

func TestUnreadableUpstreamResponseIsNotSuccess(t *testing.T) {
	router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("private upstream diagnostic"))
	})
	response := postJSON(router, "/api/explain", `{"features":{}}`)
	if response.Code != 502 || strings.Contains(response.Body.String(), "private") {
		t.Fatalf("unsafe response: %d %s", response.Code, response.Body)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return fn(r) }

func TestUpstreamTimeoutIsExplicit(t *testing.T) {
	router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("test should use the injected transport")
	})
	pythonClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		return nil, context.DeadlineExceeded
	})}
	if response := postJSON(router, "/api/explain", `{"features":{}}`); response.Code != 504 {
		t.Fatalf("status = %d; want 504", response.Code)
	}
}

func TestForwardingPreservesRequestCancellation(t *testing.T) {
	pythonStub(t, func(w http.ResponseWriter, r *http.Request) {})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	pythonClient = &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Context().Err() != context.Canceled {
			t.Error("caller cancellation was not propagated")
		}
		return nil, r.Context().Err()
	})}
	if _, err := forwardToPython(ctx, "/explain", PredictRequest{}); err == nil {
		t.Error("expected cancelled request")
	}
}

func TestCompareDoesNotPanicOnUpstreamError(t *testing.T) {
	router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(503)
		w.Write([]byte(`{"detail":"Models are not loaded"}`))
	})
	response := postJSON(router, "/api/compare", `{"coffeeA":{"features":{}},"coffeeB":{"features":{}}}`)
	if response.Code != 503 {
		t.Fatalf("status = %d; want 503", response.Code)
	}
}

func TestCompareStillReturnsScoreDifference(t *testing.T) {
	calls := 0
	router := pythonStub(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		json.NewEncoder(w).Encode(map[string]float64{"score": 80 + float64(calls)})
	})
	response := postJSON(router, "/api/compare", `{"coffeeA":{"features":{}},"coffeeB":{"features":{}}}`)
	var result struct {
		Diff float64 `json:"diff"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if response.Code != 200 || result.Diff != -1 {
		t.Fatalf("comparison changed: %d %s", response.Code, response.Body)
	}
}
