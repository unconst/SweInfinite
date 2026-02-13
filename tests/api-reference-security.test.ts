import request from 'supertest';
import { app } from '../src/app';

describe('API Reference Security', () => {
  it('should not expose API_REFERENCE.md as a static file', async () => {
    const response = await request(app).get('/API_REFERENCE.md');
    
    expect(response.status).toBe(404);
    expect(response.text).not.toContain('PlaySync API Endpoints Quick Reference');
    expect(response.text).not.toContain('Authentication Module');
  });
});
