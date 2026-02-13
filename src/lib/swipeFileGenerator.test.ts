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

describe('Anthropic model ID usage in swipeFileGenerator', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockMessagesCreate.mockResolvedValue({
      content: [{ type: 'text', text: '[]' }],
    });
  });

  it('should use claude-opus-4-6 (not dated variant) when generating swipe files', async () => {
    const { generateSwipeFile } = await import('./swipeFileGenerator');
    
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
      await generateSwipeFile('test-user-id', mockContext, 5);
    } catch (error) {
      // Ignore execution errors, we only care about the model parameter
    }

    expect(mockMessagesCreate).toHaveBeenCalled();
    const callArgs = mockMessagesCreate.mock.calls[0][0];
    expect(callArgs.model).toBe('claude-opus-4-6');
    expect(callArgs.model).not.toBe('claude-opus-4-6-20260204');
  });
});
