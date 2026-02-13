import { renderHook, waitFor } from '@testing-library/react';
import { vi, describe, it, expect } from 'vitest';
import { useState, useEffect } from 'react';

describe('Auto-publish attestation effect', () => {
  it('should automatically call publishAttestationToServer when attestationResult changes from null to a value and publishStatus is idle', async () => {
    const mockPublishAttestationToServer = vi.fn();
    const publishStatus = 'idle';

    const { result, rerender } = renderHook(() => {
      const [attestationResult, setAttestationResult] = useState<{
        attestation: string;
        frame_hash: string;
        execution_context_hash: string;
        expires_at: string;
      } | null>(null);

      useEffect(() => {
        if (attestationResult && publishStatus === 'idle') {
          mockPublishAttestationToServer();
        }
      }, [attestationResult]);

      return { attestationResult, setAttestationResult };
    });

    expect(mockPublishAttestationToServer).not.toHaveBeenCalled();

    result.current.setAttestationResult({
      attestation: 'eyJwYXlsb2FkIjp7ImdhdGVfY29udGVudF9oYXNoZXMiOnsiZm9vIjoiYmFyIn19fQ==',
      frame_hash: 'sha256:abc123',
      execution_context_hash: 'sha256:def456',
      expires_at: '2024-01-01T00:00:00Z',
    });

    rerender();

    await waitFor(() => {
      expect(mockPublishAttestationToServer).toHaveBeenCalledTimes(1);
    });
  });

  it('should not call publishAttestationToServer when publishStatus is not idle', async () => {
    const mockPublishAttestationToServer = vi.fn();
    const publishStatus = 'publishing';

    const { result, rerender } = renderHook(() => {
      const [attestationResult, setAttestationResult] = useState<{
        attestation: string;
        frame_hash: string;
        execution_context_hash: string;
        expires_at: string;
      } | null>(null);

      useEffect(() => {
        if (attestationResult && publishStatus === 'idle') {
          mockPublishAttestationToServer();
        }
      }, [attestationResult]);

      return { attestationResult, setAttestationResult };
    });

    result.current.setAttestationResult({
      attestation: 'test-attestation',
      frame_hash: 'test-frame-hash',
      execution_context_hash: 'test-execution-hash',
      expires_at: '2024-01-01T00:00:00Z',
    });

    rerender();

    expect(mockPublishAttestationToServer).not.toHaveBeenCalled();
  });
});
