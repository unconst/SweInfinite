import Anthropic from '@anthropic-ai/sdk';

// Mock Anthropic before importing the module
const mockMessagesCreate = jest.fn();
jest.mock('@anthropic-ai/sdk', () => {
  return jest.fn().mockImplementation(() => ({
    messages: {
      create: mockMessagesCreate,
    },
  }));
});

// Set environment variable before importing
process.env.CLAUDE_API_KEY = 'test-api-key';

describe('Anthropic model ID usage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockMessagesCreate.mockResolvedValue({
      content: [{ type: 'text', text: '{"ideas": []}' }],
    });
  });

  it('should use claude-opus-4-6 (not dated variant) when generating ideas', async () => {
    const { generateIdeas } = await import('./scriptGenerator');
    
    const mockContext = {
      niche: 'test',
      targetAudience: 'test',
      contentGoal: 'test',
      statusProof: [],
      powerExamples: [],
      credibilityMarkers: [],
      likenessTraits: [],
      toneOfVoice: 'test',
      personalCatchphrases: [],
      avoidTopics: [],
      exampleScripts: [],
      primaryPlatform: 'test',
      typicalVideoLength: 60,
    };

    try {
      await generateIdeas('test-user-id', mockContext, 'auto');
    } catch (error) {
      // Ignore execution errors, we only care about the model parameter
    }

    expect(mockMessagesCreate).toHaveBeenCalled();
    const callArgs = mockMessagesCreate.mock.calls[0][0];
    expect(callArgs.model).toBe('claude-opus-4-6');
    expect(callArgs.model).not.toBe('claude-opus-4-6-20260204');
  });

  it('should use claude-opus-4-6 (not dated variant) when generating scripts', async () => {
    const { generateScript } = await import('./scriptGenerator');
    
    const mockIdea = {
      title: 'test',
      hook: 'test',
      hookVariations: [],
      angle: 'test',
      contentType: 'reach' as const,
      engagementPlay: 'saves' as const,
      spclElements: {
        status: 'test',
        power: 'test',
        credibility: 'test',
        likeness: 'test',
      },
      targetEmotion: 'test',
      estimatedLength: 60,
    };

    try {
      await generateScript('test-user-id', mockIdea, 'auto');
    } catch (error) {
      // Ignore execution errors, we only care about the model parameter
    }

    expect(mockMessagesCreate).toHaveBeenCalled();
    const callArgs = mockMessagesCreate.mock.calls[0][0];
    expect(callArgs.model).toBe('claude-opus-4-6');
    expect(callArgs.model).not.toBe('claude-opus-4-6-20260204');
  });
});
