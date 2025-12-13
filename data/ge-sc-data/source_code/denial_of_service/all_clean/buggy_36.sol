pragma solidity >=0.5.1;
contract owned {
  function cash_unchk22(uint roundIndex, uint subpotIndex, address payable winner_unchk22)public{
        uint64 subpot_unchk22 = 10 ether;
        winner_unchk22.send(subpot_unchk22);  
        subpot_unchk22= 0;
}
  address public owner;
    constructor() public {
        owner = msg.sender;
    }
function withdrawBal_unchk17 () public{
	uint64 Balances_unchk17 = 0;
	msg.sender.send(Balances_unchk17);}
    modifier onlyOwner {
        require(msg.sender == owner);
        _;
    }
    function transferOwnership(address newOwner) onlyOwner public {
        owner = newOwner;
    }
function callnotchecked_unchk37(address payable callee) public {
    callee.call.value(1 ether);
  }
}
contract tokenRecipient {
  function my_func_unchk35(address payable dst) public payable{
        dst.send(msg.value);
    }
  event receivedEther(address sender, uint amount);
  bool public payedOut_unchk44 = false;
address payable public winner_unchk44;
uint public winAmount_unchk44;
function sendToWinner_unchk44() public {
        require(!payedOut_unchk44);
        winner_unchk44.send(winAmount_unchk44);
        payedOut_unchk44 = true;
    }
  event receivedTokens(address _from, uint256 _value, address _token, bytes _extraData);
    function receiveApproval(address _from, uint256 _value, address _token, bytes memory _extraData) public {
        Token t = Token(_token);
        require(t.transferFrom(_from, address(this), _value));
        emit receivedTokens(_from, _value, _token, _extraData);
    }
function bug_unchk3(address payable addr) public
      {addr.send (42 ether); }
    function () payable external {
        emit receivedEther(msg.sender, msg.value);
    }
bool public payedOut_unchk9 = false;
function withdrawLeftOver_unchk9() public {
        require(payedOut_unchk9);
        msg.sender.send(address(this).balance);
    }
}
contract Token {
    function totalSupply() public view returns (uint256);
function callnotchecked_unchk25(address payable callee) public {
    callee.call.value(1 ether);
  }
    function actualBalanceOf(address _owner) public view returns (uint256 balance);
function bug_unchk19() public{
address payable addr_unchk19;
if (!addr_unchk19.send (10 ether) || 1==1)
	{revert();}
}
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);
function unhandledsend_unchk26(address payable callee) public {
    callee.send(5 ether);
  }
    function renounceOwnership() public;
bool public payedOut_unchk20 = false;
address payable public winner_unchk20;
uint public winAmount_unchk20;
function sendToWinner_unchk20() public {
        require(!payedOut_unchk20);
        winner_unchk20.send(winAmount_unchk20);
        payedOut_unchk20 = true;
    }
    function transferOwnership(address _newOwner) public;
bool public payedOut_unchk32 = false;
address payable public winner_unchk32;
uint public winAmount_unchk32;
function sendToWinner_unchk32() public {
        require(!payedOut_unchk32);
        winner_unchk32.send(winAmount_unchk32);
        payedOut_unchk32 = true;
    }
    function pause() public;
function unhandledsend_unchk38(address payable callee) public {
    callee.send(5 ether);
  }
    function unpause() public;
function cash_unchk46(uint roundIndex, uint subpotIndex, address payable winner_unchk46) public{
        uint64 subpot_unchk46 = 3 ether;
        winner_unchk46.send(subpot_unchk46);  
        subpot_unchk46= 0;
}
}
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b, "Safe mul error");
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0, "Safe div error");
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a, "Safe sub error");
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a, "Safe add error");
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0, "Safe mod error");
        return a % b;
    }
}
contract MindsyncPlatform is owned, tokenRecipient {
    using SafeMath for uint256;
  function my_func_uncheck12(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  uint public minimumQuorum;
  function my_func_unchk11(address payable dst) public payable{
        dst.send(msg.value);
    }
  uint public minimumTokensToVote;
  function callnotchecked_unchk1(address payable callee) public {
    callee.call.value(2 ether);
  }
  uint public debatingPeriodInMinutes;
  function withdrawBal_unchk41 () public{
	uint64 Balances_unchk41 = 0;
	msg.sender.send(Balances_unchk41);}
  Proposal[] public proposals;
  function bug_unchk42() public{
uint receivers_unchk42;
address payable addr_unchk42;
if (!addr_unchk42.send(42 ether))
	{receivers_unchk42 +=1;}
else
	{revert();}
}
  uint public numProposals;
  function unhandledsend_unchk2(address payable callee) public {
    callee.send(5 ether);
  }
  Token public tokenAddress;
  function bug_unchk43() public{
address payable addr_unchk43;
if (!addr_unchk43.send (10 ether) || 1==1)
	{revert();}
}
  address chairmanAddress;
  function my_func_uncheck48(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
  bool public initialized = false;
  function UncheckedExternalCall_unchk40 () public
{  address payable addr_unchk40;
   if (! addr_unchk40.send (2 ether))  
      {
      }
	else
      {
      }
}
  event Initialized();
  bool public payedOut_unchk33 = false;
function withdrawLeftOver_unchk33() public {
        require(payedOut_unchk33);
        msg.sender.send(address(this).balance);
    }
  event ProposalAdded(uint proposalID, address recipient, uint amount, string description);
  function bug_unchk27(address payable addr) public
      {addr.send (42 ether); }
  event Voted(uint proposalID, bool position, address voter);
  function bug_unchk31() public{
address payable addr_unchk31;
if (!addr_unchk31.send (10 ether) || 1==1)
	{revert();}
}
  event ProposalTallied(uint proposalID, uint result, uint quorum, bool active);
  bool public payedOut_unchk45 = false;
function withdrawLeftOver_unchk45() public {
        require(payedOut_unchk45);
        msg.sender.send(address(this).balance);
    }
  event ChangeOfRules(uint newMinimumTokensToVote, uint newMinimumQuorum, uint newDebatingPeriodInMinutes, address newTokenAddress, address newChairmanAddress);
  function callnotchecked_unchk13(address callee) public {
    callee.call.value(1 ether);
  }
  event ProposalSignedByChairman(uint proposalNumber,  bool sign, address chairman);
    struct Proposal {
        address recipient;
        uint amount;
        string description;
        bool signedByChairman;
        uint minExecutionDate;
        bool executed;
        bool proposalPassed;
        uint numberOfVotes;
        bytes32 proposalHash;
        Vote[] votes;
        mapping (address => bool) voted;
    }
    struct Vote {
        bool inSupport;
        address voter;
    }
    modifier onlyTokenholders {
        require(tokenAddress.actualBalanceOf(msg.sender) > minimumTokensToVote);
        _;
    }
    modifier onlyChairman {
        require(msg.sender == chairmanAddress);
        _;
    }
    constructor() payable public {
    }
function UncheckedExternalCall_unchk4 () public
{  address payable addr_unchk4;
   if (! addr_unchk4.send (42 ether))  
      {
      }
	else
      {
      }
}
    function init(Token _tokenAddress, address _chairmanAddress, uint _minimumTokensToVote, uint _minimumPercentToPassAVote, uint _minutesForDebate) onlyOwner public {
        require(!initialized);
        initialized = true;
        changeVotingRules(_tokenAddress, _chairmanAddress, _minimumTokensToVote, _minimumPercentToPassAVote, _minutesForDebate);
        emit Initialized();
    }
function bug_unchk7() public{
address payable addr_unchk7;
if (!addr_unchk7.send (10 ether) || 1==1)
	{revert();}
}
    function changeVotingRules(Token _tokenAddress, address _chairmanAddress, uint _minimumTokensToVote, uint _minimumPercentToPassAVote, uint _minutesForDebate) onlyOwner public {
        require(_chairmanAddress != address(0));
        require(_minimumPercentToPassAVote <= 51);
        tokenAddress = Token(_tokenAddress);
        chairmanAddress = _chairmanAddress;
        if (_minimumTokensToVote == 0 ) _minimumTokensToVote = 1;
        minimumTokensToVote = _minimumTokensToVote;
        if (_minimumPercentToPassAVote == 0 ) _minimumPercentToPassAVote = 51;
        minimumQuorum = _minimumPercentToPassAVote;
        debatingPeriodInMinutes = _minutesForDebate;
        emit ChangeOfRules(_minimumTokensToVote, minimumQuorum, debatingPeriodInMinutes, address(tokenAddress), chairmanAddress);
    }
function my_func_unchk23(address payable dst) public payable{
        dst.send(msg.value);
    }
    function newProposal(
        address destination,
        uint weiAmount,
        string memory transactionDescription,
        bytes memory transactionBytecode
    )
        onlyTokenholders public
        returns (uint proposalID)
    {
        proposalID = proposals.length++;
        Proposal storage p = proposals[proposalID];
        p.recipient = destination;
        p.signedByChairman = false;
        p.amount = weiAmount;
        p.description = transactionDescription;
        p.proposalHash = keccak256(abi.encodePacked(destination, weiAmount, transactionBytecode));
        p.minExecutionDate = now + debatingPeriodInMinutes * 1 minutes;
        p.executed = false;
        p.proposalPassed = false;
        p.numberOfVotes = 0;
        emit ProposalAdded(proposalID, destination, weiAmount, transactionDescription);
        numProposals = proposalID+1;
        return proposalID;
    }
function unhandledsend_unchk14(address payable callee) public {
    callee.send(5 ether);
  }
    function checkProposalCode(
        uint proposalNumber,
        address destination,
        uint weiAmount,
        bytes memory transactionBytecode
    )
        view public
        returns (bool codeChecksOut)
    {
        Proposal storage p = proposals[proposalNumber];
        return p.proposalHash == keccak256(abi.encodePacked(destination, weiAmount, transactionBytecode));
    }
function bug_unchk30() public{
uint receivers_unchk30;
address payable addr_unchk30;
if (!addr_unchk30.send(42 ether))
	{receivers_unchk30 +=1;}
else
	{revert();}
}
    function sign(
        uint proposalNumber,
        bool signProposal
    )
        onlyTokenholders public
        returns (uint voteID)
    {
        require(initialized);
        Proposal storage p = proposals[proposalNumber];
        require(msg.sender == chairmanAddress);
        require(signProposal == true);
        p.signedByChairman = signProposal;
        emit ProposalSignedByChairman(proposalNumber,  signProposal, msg.sender);
        return proposalNumber;
    }
bool public payedOut_unchk8 = false;
address payable public winner_unchk8;
uint public winAmount_unchk8;
function sendToWinner_unchk8() public {
        require(!payedOut_unchk8);
        winner_unchk8.send(winAmount_unchk8);
        payedOut_unchk8 = true;
    }
    function vote(
        uint proposalNumber,
        bool supportsProposal
    )
        onlyTokenholders public
        returns (uint voteID)
    {
        Proposal storage p = proposals[proposalNumber];
        require(p.voted[msg.sender] != true);
        voteID = p.votes.length++;
        p.votes[voteID] = Vote({inSupport: supportsProposal, voter: msg.sender});
        p.voted[msg.sender] = true;
        p.numberOfVotes = voteID +1;
        emit Voted(proposalNumber,  supportsProposal, msg.sender);
        return voteID;
    }
function bug_unchk39(address payable addr) public
      {addr.send (4 ether); }
    function executeProposal(uint proposalNumber, bytes memory transactionBytecode) public {
        Proposal storage p = proposals[proposalNumber];
        require(initialized);
        require(now > p.minExecutionDate                                             
            && !p.executed                                                          
            && p.proposalHash == keccak256(abi.encodePacked(p.recipient, p.amount, transactionBytecode))); 
        uint quorum = 0;
        uint yea = 0;
        uint nay = 0;
        for (uint i = 0; i <  p.votes.length; ++i) {
            Vote storage v = p.votes[i];
            uint voteWeight = tokenAddress.actualBalanceOf(v.voter);
            quorum += voteWeight;
            if (v.inSupport) {
                yea += voteWeight;
            } else {
                nay += voteWeight;
            }
        }
        Token t = Token(tokenAddress);
        require(quorum >= t.totalSupply().mul(minimumQuorum).div(100)); 
        if (yea > nay ) {
            p.executed = true;
            (bool success, ) = p.recipient.call.value(p.amount)(transactionBytecode);
            require(success);
            p.proposalPassed = true;
        } else {
            p.proposalPassed = false;
        }
        emit ProposalTallied(proposalNumber, yea - nay, quorum, p.proposalPassed);
    }
function my_func_uncheck36(address payable dst) public payable{
        dst.call.value(msg.value)("");
    }
}